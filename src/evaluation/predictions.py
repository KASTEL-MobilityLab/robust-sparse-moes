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
from src.models.nn.attacks import AttackModule
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


def _setup(config):
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, use_clearml=False)
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

    def _load_state_dict(run_name):
        runs = get_runs(project=wandb.run.project, name=run_name)

        for i, run in enumerate(runs):
            checkpoint_id = f"model-{run.id}"
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
        state_dict = ckpt["state_dict"]

        stripped_sd = {key.replace("model.model", "model"): val for key, val in state_dict.items()}

        return stripped_sd

    assert "run_name" in config.wandb, "Model checkpoint missing."
    model.load_state_dict(_load_state_dict(config.wandb.run_name))

    metric = hydra.utils.instantiate(config.metric)

    normalization = hydra.utils.instantiate(config.normalization)
    inverse_normalization = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0], std=[1 / x for x in normalization.std]),
            transforms.Normalize(mean=[-x for x in normalization.mean], std=[1.0]),
        ]
    )

    return datamodule, model, metric, normalization, inverse_normalization


def _instantiate_attack_modules(
    config, model, normalization
) -> Tuple[List[str], List[float], List[AttackModule]]:
    EPSLIST = False
    if EPSLIST:
        eps_list, alpha, steps = config.attack.epsilon, config.attack.alpha, config.attack.steps
        combis = [(steps, eps, alpha) for eps in eps_list]
        severity = eps_list
    else:
        eps, alpha, steps_list = config.attack.epsilon, config.attack.alpha, config.attack.steps
        combis = [(steps, eps, alpha) for steps in steps_list]
        severity = steps_list

    partial_attacks = (
        partial(attack.pgd, eps=eps / 255, alpha=alpha / 255, steps=steps, random_start=False)
        for steps, eps, alpha in combis
    )

    attack_modules = [
        AttackModule(model, attack=partial_attack, normalization=normalization)
        for partial_attack in partial_attacks
    ]
    names = [f"PGD-{steps}-{eps}-{alpha}" for steps, eps, alpha in combis]

    return names, severity, attack_modules


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


def _run_model_on_dataset(
    dataloader: DataLoader,
    model: torch.nn.Module,
    attack_model: AttackModule,
    tracker: PerformanceTracker,
    inverse_normalization: torch.nn.Module,
) -> None:
    # output_queue = queue.Queue(maxsize=10)
    # output_task_done = False
    # def process_outputs():
    #    id =0
    #    while not output_task_done:
    #        print(f"Process batch {id}")
    #        try:
    #            output = output_queue.get()
    #            tracker.process_batch_output(*output)
    #            output_queue.task_done()
    #            id +=1
    #        except Exception as e:
    #            print(e)
    #            raise e

    # output_thread = Thread(target=process_outputs)
    # output_thread.daemon = True
    # output_thread.start()

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = batch
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        if attack_model is not None:
            attacked_inputs, _ = attack_model(model, batch=(inputs, targets))
        else:
            attacked_inputs = inputs

        losses, logits, preds = _run_batch(model, (attacked_inputs, targets))

        # Need to reverse the normalization!
        inputs = inverse_normalization(inputs)
        attacked_inputs = inverse_normalization(attacked_inputs)

        # output_queue.put((inputs, attacked_inputs, targets, losses, preds))
        tracker.process_batch_output(inputs, attacked_inputs, targets, losses, preds)


def _run_dataset(
    dataloader: DataLoader,
    model: torch.nn.Module,
    attack_modules: List[AttackModule],
    metric: torchmetrics.Metric,
    inverse_normalization: torch.nn.Module,
    random_samples: List[int],
    max_queue_size=8,
):
    attack_models = [None, *attack_modules]

    metric.reset()
    metrics = [copy.deepcopy(metric) for _ in attack_models]

    trackers = [
        PerformanceTracker(
            metric,
            worst_sample_queue=PriorityQueue(maxsize=max_queue_size + 1),
            random_sample_indices=random_samples,
        )
        for metric in metrics
    ]

    for tracker, attack_model in zip(trackers, attack_models):
        _run_model_on_dataset(dataloader, model, attack_model, tracker, inverse_normalization)

    return trackers


def _to_img(tens) -> np.array:
    return (tens.clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


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
    attack_names: List[str],
    attack_severity: List[float],
    class_map: dict,
    trackers: List[PerformanceTracker],
):
    data = []
    for steps, t in zip([0, *attack_severity], trackers):
        result = t.metric.compute().item()
        loss = torch.tensor(t.losses).mean()
        data.append((steps, result, loss))

    table = wandb.Table(data=data, columns=["PGD-Steps", "Metric", "Loss"])
    wandb.log(
        {
            "pgd_performance_plot": wandb.plot.line(
                table, "PGD-Steps", "Metric", title="Performance versus PGD-Steps"
            ),
            "pgd_loss_plot": wandb.plot.line(
                table, "PGD-Steps", "Loss", title="Loss versus PGD-Steps"
            ),
            **{f"pgd_{steps}_metric": perf for steps, perf, _ in data},
            **{f"pgd_{steps}_loss": loss for steps, _, loss in data},
        }
    )

    columns = [
        "Raw image",
        *attack_names,
        *(f"Difference Img: {steps}" for steps in attack_names),
        "Target",
        *(f"Prediction: {steps}" for steps in [0, *attack_names]),
    ]

    sample_sets = zip(*(tracker.random_samples for tracker in trackers))

    data = []
    for samples in sample_sets:
        samples: List[Sample] = samples

        raw_img = _to_img(samples[0].input)
        model_inputs = [_to_img(sample.model_input) for sample in samples[1:]]
        differences = [_diff_image(raw_img, model_input) for model_input in model_inputs]
        model_inputs = [wandb.Image(x) for x in model_inputs]
        raw_img = wandb.Image(raw_img)

        preds = [class_map[sample.pred.item()] for sample in samples]

        data.append(
            (raw_img, *model_inputs, *differences, class_map[samples[0].target.item()], *preds)
        )

    table = wandb.Table(
        data=data,
        columns=columns,
    )
    wandb.log({"prediction_table": table})


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

    datamodule, model, metric, normalization, inverse_normalization = _setup(config)
    dataloader = datamodule.test_dataloader()

    # natural_attack_modules = _instantiate_attack_modules(config, natural_model, normalization)
    attack_names, attack_severity, attack_modules = _instantiate_attack_modules(
        config, model, normalization
    )

    random_samples = random.sample(
        population=list(range(len(dataloader.dataset))), k=config.max_queue_size
    )
    trackers = _run_dataset(
        dataloader,
        model,
        attack_modules,
        metric,
        inverse_normalization,
        random_samples,
        max_queue_size=config.max_queue_size,
    )

    _log_results(attack_names, attack_severity, datamodule.class_map, trackers)

    # Make sure everything closed properly
    log.info("Finalizing!")
    wandb.finish()

    # Return metric score for hyperparameter optimization
