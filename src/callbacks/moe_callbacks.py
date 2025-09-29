import dataclasses
from collections import defaultdict
from functools import partial
from typing import Any, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from src import utils
from src.callbacks.wandb_callbacks import get_wandb_logger
from src.models.nn.moe.gate.simple_gate import AbstractGate

log = utils.get_logger(__name__)


@dataclasses.dataclass
class ActivationTracker:
    """Track which experts are activated."""

    prefix: str
    active: bool = False
    gate_type: Type[nn.Module] = AbstractGate
    activations: dict = dataclasses.field(default_factory=partial(defaultdict, list))
    hooks: dict = dataclasses.field(default_factory=dict)
    module_ks: dict = dataclasses.field(default_factory=dict)

    def __hash__(self):
        return hash(self.prefix)

    def _register_moe_hooks(self, model, gate_type):

        gating_layers = {n: m for n, m in model.named_modules() if isinstance(m, gate_type)}

        for name, module in gating_layers.items():

            def _register_last_moe(module, input, output, *, name):
                if self.active:
                    self.activations[name].append(tuple(out.detach().cpu() for out in output))

            self.hooks[name] = module.register_forward_hook(partial(_register_last_moe, name=name))
            self.module_ks[name] = module.k

    def _unregister_moe_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

    def _log_activations(self, scores, activations, k, name) -> dict:
        num_experts = activations.shape[1]

        # Actually activated experts in the sparse setting!
        activated_experts = torch.topk(activations, k=k, dim=1).indices
        activated_experts = activated_experts.cpu().numpy()

        hist_activations = wandb.Histogram(np_histogram=np.histogram(scores, range=(0, 1)))

        np_histogram = np.histogram(activated_experts, bins=num_experts)
        hist = wandb.Histogram(np_histogram=np_histogram)

        return {
            f"{self.prefix}_expert_activation_scores/{name}": hist_activations,
            f"{self.prefix}_selected_experts/{name}": hist,
        }

    def _create_plots(self, experiment):

        module_names = list(self.hooks.keys())

        log_dict = {}
        for name in module_names:
            if len(self.activations[name]) == 0:
                print("empty activations")
                # logger.warning("Empty activations")
                continue
            scores = torch.cat(list(x for x, _ in self.activations[name]), dim=0)
            activations = torch.cat(list(x for _, x in self.activations[name]), dim=0)

            log_dict.update(
                self._log_activations(scores, activations, k=self.module_ks[name], name=name)
            )

        wandb.log(log_dict)

    def activate(self, module):
        self.active = True
        self._register_moe_hooks(module, self.gate_type)

    def deactivate(self, trainer):

        if not self.active or len(self.hooks) == 0:
            return
        self.active = False

        logger = get_wandb_logger(trainer)
        experiment = logger.experiment
        self._create_plots(experiment)

        self.activations.clear()

        self._unregister_moe_hooks()


@dataclasses.dataclass
class ActivationTargetTracker(ActivationTracker):
    targets: list = dataclasses.field(default_factory=list)

    def __hash__(self):
        return hash(self.prefix)

    def deactivate(self, trainer):
        super().deactivate(trainer)
        self.targets.clear()

    @staticmethod
    def _create_plot(activations, targets):
        corr = np.zeros((targets.max() + 2, activations.shape[1]))
        for t, a in zip(targets, activations):
            corr[t] += a
        corr[-1] = corr.sum(axis=0)

        # breakpoint()
        # corr /= corr.sum(axis=0)
        # Possibly the ugliest way to normalize by total number of activations of expert
        corr = np.round((corr.T / corr.T.sum(axis=0)).T, decimals=3)

        # set figure size
        plt.figure(figsize=(14, 8))
        plt.xlabel("Experts")
        plt.ylabel("Targets")

        # set labels size
        sn.set(font_scale=1.4)

        # set font size
        df = pd.DataFrame(
            corr,
            columns=[f"{i}" for i in range(corr.shape[1])],
            index=[f"{i}" for i in range(corr.shape[0] - 1)] + ["Total"],
        )
        df.index.name = "Targets"
        df.columns.name = "Experts"

        sn.heatmap(df, annot=True, annot_kws={"size": 8}, fmt="g")

    def _create_plots(self, experiment):

        module_names = list(self.hooks.keys())

        targets = torch.cat(self.targets).cpu().numpy()
        for name in module_names:
            if len(self.activations[name]) == 0:
                print("empty activations")
                # logger.warning("Empty activations")
                continue
            activations = (
                torch.cat(list(x for _, x in self.activations[name]), dim=0).cpu().numpy()
            )

            self._create_plot(activations, targets)

            # names should be unique or else charts from different experiments in wandb will overlap
            experiment.log(
                {f"{self.prefix}_expert_correlation_matrix/{name}": wandb.Image(plt)},
                commit=False,
            )

            # reset plot
            plt.clf()


class LogExpertLabelCorrelationMatrix(Callback):
    """Generate confusion matrix between experts and labels.

    Expects validation step to return predictions and targets.
    """

    def __init__(self, gate_type=AbstractGate):
        self.ready = True
        self.train_tracker = ActivationTargetTracker("train", gate_type=gate_type)
        self.val_tracker = ActivationTargetTracker("val", gate_type=gate_type)
        self.test_tracker = ActivationTargetTracker("test", gate_type=gate_type)
        self.all_trackers = {self.train_tracker, self.val_tracker, self.test_tracker}

    def _turn_on(self, tracker: ActivationTracker, pl_module, trainer):
        if not self.ready:
            return
        trackers = self.all_trackers.copy()
        trackers.remove(tracker)

        for tr in trackers:
            tr.deactivate(trainer)

        tracker.activate(pl_module)

    def _turn_off_all(self, trainer):
        trackers = {self.train_tracker, self.val_tracker, self.test_tracker}

        for tr in trackers:
            tr.deactivate(trainer)

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False
        self._turn_off_all(trainer)

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def _on_batch_end(self, outputs):
        """Gather data from single batch."""
        for tr in self.all_trackers:
            if tr.active:
                tr.targets.append(outputs["targets"])

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._turn_on(self.val_tracker, pl_module, trainer)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self._on_batch_end(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        self._turn_off_all(trainer)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._turn_on(self.test_tracker, pl_module, trainer)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._on_batch_end(outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        self._turn_off_all(trainer)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        # Only report 10 batches!
        if batch_idx > 10:
            self._turn_off_all(trainer)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._turn_on(self.train_tracker, pl_module, trainer)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self._on_batch_end(outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        self._turn_off_all(trainer)


class LogExpertActivation(Callback):
    """Track individual expert activations."""

    def __init__(self, gate_type=AbstractGate):
        self.ready = True
        self.train_tracker = ActivationTracker("train", gate_type=gate_type)
        self.val_tracker = ActivationTracker("val", gate_type=gate_type)
        self.test_tracker = ActivationTracker("test", gate_type=gate_type)
        self.all_trackers = {self.train_tracker, self.val_tracker, self.test_tracker}

    def _turn_on(self, tracker: ActivationTracker, pl_module, trainer):
        if not self.ready:
            return
        trackers = self.all_trackers.copy()
        trackers.remove(tracker)

        for tr in trackers:
            tr.deactivate(trainer)

        tracker.activate(pl_module)

    def _turn_off_all(self, trainer):
        trackers = {self.train_tracker, self.val_tracker, self.test_tracker}

        for tr in trackers:
            tr.deactivate(trainer)

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False
        self._turn_off_all(trainer)

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._turn_on(self.val_tracker, pl_module, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        self._turn_off_all(trainer)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._turn_on(self.test_tracker, pl_module, trainer)

    def on_test_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        self._turn_off_all(trainer)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._turn_on(self.train_tracker, pl_module, trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        self._turn_off_all(trainer)
