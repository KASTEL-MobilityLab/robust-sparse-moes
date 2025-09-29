"""Produce evaluation of best/worst samples w.r.t. a set of attacks.

Attacks:
* PGD
* vary number of steps
* adapt alpha (step size) inverse proportional to num steps
* e.g. 20,40,60

Metric:
* Calculate a global metric on the dataset for each attack to quantify robustness

Outputs (probably as logging output to W&B):
* {Worst, Random, Best} samples
* Corresponding outputs
*
"""
import copy
import itertools
import random
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from queue import PriorityQueue
from typing import List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, seed_everything
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src import utils
from src.callbacks.cityscapes_callbacks import wb_mask
from src.models.nn.attacks import AttackModule
from src.models.nn.moe.layer import MOELayer
from src.models.nn.moe.resnet_block_moe import MoEModel
from src.utils import attack, flatten_dict
from src.utils.wandb import get_runs, load_checkpoint_file

log = utils.get_logger(__name__)


# Allow partials in hydra configs
def get_method(method_args):
    def new_method(*args, **kwargs):
        if "_name_" in kwargs:
            del kwargs["_name_"]
        method = hydra.utils.get_method(method_args)
        return method(*args, **kwargs)

    return new_method


OmegaConf.register_new_resolver("get_method", get_method)


class Task(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


def _load_state_dict(run_name, task=Task.SEGMENTATION):
    print("Loading project", wandb.run.project)
    print("Loading run", run_name)
    runs = get_runs(project=wandb.run.project, name=run_name)

    for i, run in enumerate(runs):
        checkpoint_id = f"model-{run.id}:v0"
        try:
            checkpoint_path = load_checkpoint_file(checkpoint_id, "model.ckpt")
            break
        except Exception:
            log.warning(
                f"There are multiple runs with this name. Could not find artifacts in the {i}th one."
            )
    else:
        raise ValueError(f"Failed to find artifacts for run {run_name}.")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    if task is Task.SEGMENTATION:
        stripped_sd = {key.replace("model.", ""): val for key, val in state_dict.items()}
    else:
        stripped_sd = {key.replace("model.model", "model"): val for key, val in state_dict.items()}

    return stripped_sd


def _setup(config):
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, use_clearml=config.get("use_clearml", False)
    )
    datamodule.prepare_data()
    datamodule.setup()

    log.info(f"Instantiating model <{config.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(config.model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    log.info("Logging config!")
    wandb.config.update(flatten_dict(config))

    # Send some parameters from config to all lightning loggers
    assert config.wandb.model_checkpoint_tag in {
        "latest",
        "best",
    }, "Unauthorized model checkpoint tag."

    task = Task(config.get("task", Task.CLASSIFICATION.value))

    assert "run_name" in config.wandb, "Model checkpoint missing."
    model.load_state_dict(_load_state_dict(config.wandb.run_name, task))

    metric = hydra.utils.instantiate(config.metric)

    normalization = hydra.utils.instantiate(config.normalization)
    inverse_normalization = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0], std=[1 / x for x in normalization.std]),
            transforms.Normalize(mean=[-x for x in normalization.mean], std=[1.0]),
        ]
    )

    return datamodule, model, metric, normalization, inverse_normalization, config.moe_layer, task


def _instantiate_attack_module(config, model, normalization) -> Tuple[str, AttackModule]:
    eps, alpha, steps = config.attack.epsilon, config.attack.alpha, config.attack.steps
    attack_type = config.attack.get("type", "pgd")  # 
    
    if attack_type.lower() == "autopgd":
        partial_attack = partial(
            attack.auto_pgd, eps=eps / 255, steps=steps
        )
        name = f"AutoPGD-{steps}-{eps}"
    else:  # 
        partial_attack = partial(
            attack.pgd, eps=eps / 255, alpha=alpha / 255, steps=steps, random_start=False
        )
        name = f"PGD-{steps}-{eps}-{alpha}"

    attack_module = AttackModule(model, attack=partial_attack, normalization=normalization)

    return name, attack_module


def _run_batch(model, batch: Tuple[Tensor, Tensor]):
    x, y = batch

    logits = model(x)
    loss = F.cross_entropy(logits, y.long(), ignore_index=255, reduction="none")
    preds = logits[0] if isinstance(logits, tuple) else logits
    preds = torch.argmax(preds, dim=1)
    return loss, logits, preds


@dataclass
class Sample:
    input: Tensor
    model_input: Tensor
    target: Tensor
    loss: Tensor
    pred: Tensor

    def __post_init__(self):
        self.input = self.input.detach().cpu()
        self.model_input = self.model_input.detach().cpu()
        self.target = self.target.detach().cpu()
        self.loss = self.loss.detach().cpu()
        self.pred = self.pred.detach().cpu()


@dataclass
class PerformanceTracker:
    metric: torchmetrics.Metric
    worst_sample_queue: PriorityQueue
    random_sample_indices: List[int]
    random_samples: List[Sample] = field(default_factory=list)
    losses: List[torch.Tensor] = field(default_factory=list)
    current_index: int = 0

    def process_sample(
        self,
        index: int,
        sample: Sample,
    ) -> None:
        # self.worst_sample_queue.put((sample.loss.item(), sample))
        # if self.worst_sample_queue.full():
        #    # maxsize is chosen one more than num samples. If it's full, one item can be thrown away
        #    self.worst_sample_queue.get(block=False)

        if index in self.random_sample_indices:
            self.random_samples.append(sample)

        self.metric.update(sample.pred.unsqueeze(0), sample.target.unsqueeze(0))
        self.losses.append(sample.loss)

    def process_batch_output(self, inputs, model_inputs, targets, losses, preds):
        samples = list(
            Sample(*item)
            for item in itertools.zip_longest(inputs, model_inputs, targets, losses, preds)
        )
        for sample in samples:
            self.process_sample(self.current_index, sample)
            self.current_index += 1

    def reduce(self):
        result = self.metric.compute().item()
        means = [loss.mean() for loss in self.losses]
        loss = torch.tensor(means).mean()
        self.data = (result, loss)
        del self.metric, self.losses


CROP_SIZE = 769
STRIDE = 512


def image_to_tiles(x: torch.Tensor, y: torch.Tensor, void_class=255):
    _, c, h, w = x.shape

    h_idx = random.randint(0, h - CROP_SIZE)
    h_idx_2 = random.randint(0, h - CROP_SIZE)
    w_idx = random.randint(0, w - 2 * CROP_SIZE)
    w_idx_2 = random.randint(w_idx, w - CROP_SIZE)
    coordinates = ((h_idx, w_idx), (h_idx_2, w_idx_2))

    batches = (
        (
            x[..., h : h + CROP_SIZE, w : w + CROP_SIZE],
            y[..., h : h + CROP_SIZE, w : w + CROP_SIZE],
        )
        for h, w in coordinates
    )

    padded = (
        (
            F.pad(x, (0, CROP_SIZE - x.shape[-1], 0, CROP_SIZE - x.shape[-2]), "constant", 0),
            F.pad(
                y, (0, CROP_SIZE - y.shape[-1], 0, CROP_SIZE - y.shape[-2]), "constant", void_class
            ),
        )
        for x, y in batches
    )

    return padded


def _run_model_on_dataset(
    dataloader: DataLoader,
    model: torch.nn.Module,
    attack_model: AttackModule,
    tracker: PerformanceTracker,
    inverse_normalization: torch.nn.Module,
    tiling=False,
) -> None:
    if tiling:
        total = 2 * len(dataloader)
        tiles = (image_to_tiles(*batch) for batch in iter(dataloader))
        dataloader = (batch for tile in tiles for batch in tile)
    else:
        total = len(dataloader)

    for idx, batch in tqdm(enumerate(dataloader), total=total):
        inputs, targets = batch
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        if attack_model is not None:
            attacked_inputs, _ = attack_model(model, batch=(inputs, targets))
            assert not (attacked_inputs == inputs).all()
        else:
            attacked_inputs = inputs

        losses, logits, preds = _run_batch(model, (attacked_inputs, targets))

        # Need to reverse the normalization!
        inputs = inverse_normalization(inputs)
        attacked_inputs = inverse_normalization(attacked_inputs)

        tracker.process_batch_output(inputs, attacked_inputs, targets, losses, preds)


def _run_dataset(
    dataloader: DataLoader,
    model: MoEModel,
    attack_module: AttackModule,
    metric: torchmetrics.Metric,
    inverse_normalization: torch.nn.Module,
    fixed_moe_layer: str,
    random_samples: List[int],
    max_queue_size=8,
):
    if fixed_moe_layer:
        moe_layer = model.get_submodule(fixed_moe_layer)
        assert isinstance(moe_layer, MOELayer), ""
        num_experts = moe_layer.num_local_experts
    else:
        num_experts = 0

    metric.reset()
    metrics = [copy.deepcopy(metric) for _ in range(num_experts + 1)]

    trackers = [
        PerformanceTracker(
            metric,
            worst_sample_queue=PriorityQueue(maxsize=max_queue_size + 1),
            random_sample_indices=random_samples,
        )
        for metric in metrics
    ]

    for i, tracker in enumerate(trackers):
        seed_everything(42)

        if fixed_moe_layer:
            moe_layer.fixed_expert = i
        _run_model_on_dataset(dataloader, model, attack_module, tracker, inverse_normalization)
        tracker.reduce()

    return trackers


def _to_img(tens) -> np.array:
    return (tens.clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _to_mask(tens) -> np.array:
    if len(tens.shape) == 3:
        tens = tens[0]
    return tens.cpu().numpy().astype(np.uint8)


def _diff_image(raw_input, attacked_input, max_eps=64):
    # if not np.all((bg_image+eps)-attacked_image >= 0):

    # diff_img = (attacked_input + eps) - raw_input

    diff_img = (raw_input - attacked_input) * (raw_input > attacked_input) + (
        attacked_input - raw_input
    ) * (raw_input < attacked_input)

    diff_img = np.sqrt(np.sum(diff_img**2, axis=2))
    return wandb.Image(
        255 / max_eps * diff_img,
    )


def _log_results(
    attack_name: str,
    trackers: List[PerformanceTracker],
    task: Task,
    class_map: dict = None,
):
    data = [(fixed_expert, *t.data) for fixed_expert, t in enumerate(trackers)]

    table = wandb.Table(data=data, columns=["Fixed Expert", "Metric", "Loss"])
    wandb.log(
        {
            f"performance_plot_{attack_name}": wandb.plot.line(
                table,
                "Fixed Expert",
                "Metric",
                title=f"({attack_name}) Performance versus fixed expert",
            ),
            f"loss_plot_{attack_name}": wandb.plot.line(
                table, "Fixed Expert", "Loss", title=f"({attack_name}) Loss versus fixed expert"
            ),
            **{
                f"pgd_{attack_name}_expert{fixed_expert}_metric": perf
                for fixed_expert, perf, _ in data
            },
            **{
                f"pgd_{attack_name}_expert{fixed_expert}_loss": loss
                for fixed_expert, _, loss in data
            },
        }
    )

    columns = ["Raw image", "Model Input"]

    if task is Task.CLASSIFICATION:
        columns.extend(
            [
                "Target",
                *(
                    f"Prediction: {attack_name}, Fixed expert {exp}"
                    for exp in range(len(trackers) - 1)
                ),
                f"Prediction: {attack_name}",
            ]
        )

    sample_sets = zip(*(tracker.random_samples for tracker in trackers))

    data = []
    for samples in sample_sets:
        samples: List[Sample] = samples

        # Take any raw image
        raw_img = _to_img(samples[0].input)
        raw_img = wandb.Image(raw_img)

        # Take last input image (i.e. with all experts)
        model_input = _to_img(samples[-1].model_input)

        if task is Task.CLASSIFICATION:
            model_input = wandb.Image(model_input)

            preds = [class_map[sample.pred.item()] for sample in samples]
            target = class_map[samples[0].target.item()]
            data.append((raw_img, model_input, target, *preds))
        elif task is Task.SEGMENTATION:
            additional_preds = (sample.pred for sample in samples[:-1])
            additional_preds = (_to_mask(pred) for pred in additional_preds)
            additional_preds = {f"fixed_exp{i}": pred for i, pred in enumerate(additional_preds)}

            model_input = wb_mask(
                model_input,
                _to_mask(samples[-1].pred),
                _to_mask(samples[0].target),
                **additional_preds,
            )
            data.append((raw_img, model_input))

        else:
            raise NotImplementedError

    table = wandb.Table(
        data=data,
        columns=columns,
    )
    wandb.log({f"{attack_name}_prediction_table": table})


def run(config: DictConfig) -> Optional[float]:
    """Contains training/testing pipeline. Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    wandb.init(
        project=config.wandb.project,
        name=config.name,
        job_type="evaluation",
        tags=config.wandb.tags,
        save_code=True,
    )

    datamodule, model, metric, normalization, inverse_normalization, fixed_moe, task = _setup(
        config
    )
    dataloader = datamodule.test_dataloader()

    attack_name, attack_module = _instantiate_attack_module(config, model, normalization)

    # Not so random ///
    random_samples = np.linspace(
        0, len(dataloader.dataset), num=config.max_queue_size, dtype=int, endpoint=False
    ).tolist()

    attacks = [("natural", None), (attack_name, attack_module)]

    for name, attack_mod in attacks:
        trackers = _run_dataset(
            dataloader,
            model,
            attack_mod,
            metric,
            inverse_normalization,
            fixed_moe,
            random_samples,
            max_queue_size=config.max_queue_size,
        )

        _log_results(name, trackers, task, getattr(datamodule, "class_map", None))

    # Make sure everything closed properly
    log.info("Finalizing!")
    wandb.finish()


if __name__ == "__main__":
    x = torch.randn(1, 3, 1024, 2024)
    res = image_to_tiles(x, x)

    print(res)
