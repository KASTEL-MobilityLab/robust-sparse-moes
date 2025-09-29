from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, List, Optional

import torch
import torchmetrics
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_warn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class CrossEntropyLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        scheduler: Optional[Callable] = None,
        criterion: Optional[Callable] = torch.nn.CrossEntropyLoss(),
        metrics: dict = {"acc": Accuracy("multiclass", num_classes=100)},
        test_metric_prefix="test/",
        max_metric_keys: set = {"val/acc"},
        return_results_on_step: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.max_metric_keys = max_metric_keys
        self._optimizer_callable = optimizer
        self._scheduler_callable = scheduler
        self.return_results_on_step = return_results_on_step

        # self.save_hyperparameters(logger=False)

        self.model = model

        # loss function
        self.criterion = criterion

        self.train_metrics = torchmetrics.MetricCollection(
            metrics=deepcopy(dict(metrics)), prefix="train/"
        )
        self.valid_metrics = torchmetrics.MetricCollection(
            metrics=deepcopy(dict(metrics)), prefix="val/"
        )
        self.test_metrics = torchmetrics.MetricCollection(
            metrics=deepcopy(dict(metrics)), prefix=test_metric_prefix
        )
        self.max_metrics = defaultdict(MaxMetric)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _collect_aux_loss(self) -> torch.Tensor:
        losses = list(mod.loss for mod in self.modules() if hasattr(mod, "loss"))
        if len(losses) == 0:
            return None
        else:
            return sum(losses)

    def partial_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.long())
        aux_loss = self._collect_aux_loss() or torch.zeros_like(loss)

        preds = logits[0] if isinstance(logits, tuple) else logits
        preds = torch.argmax(preds, dim=1)

        return x, preds, y, loss, aux_loss

    def step(self, batch: Any, metrics: torchmetrics.Metric, prefix=""):
        _, preds, targets, loss, aux_loss = self.partial_step(batch)

        metrics.update(preds, targets)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        output = {
            prefix + "loss": loss + aux_loss,
            prefix + "main_loss": loss,
            prefix + "aux_loss": aux_loss,
        }
        if self.return_results_on_step:
            output.update(
                {
                    "preds": preds.detach().cpu(),
                    "targets": targets.detach().cpu(),
                }
            )
        return output

    def _log_metrics(self, metrics) -> None:
        """Logs all of the key value pairs in `metrics`."""
        results = metrics.compute()

        def _log(key, value):
            if key in self.max_metric_keys:
                max_metric = self.max_metrics[key]
                max_metric.update(value)
                self.log("max/" + key, max_metric.compute(), sync_dist=True)

            self.log(key, value, sync_dist=True)

        for metric_name, result in results.items():
            key = metric_name
            value_tensor = result.double().detach()
            if value_tensor.size() == torch.Size([]):
                value = value_tensor.item()
                _log(key, value)

            else:
                value = value_tensor.tolist()
                for i, v in enumerate(value):
                    _log(key + f"_id_{i}", v)

    def _epoch_end(
        self,
        outputs: Any,
        metrics: torchmetrics.MetricCollection,
        loss_prefix: Optional[str] = "",
    ):
        """Steps after an epoch."""
        self._log_metrics(metrics)
        metrics.reset()

        def log_loss(postfix):
            loss_name = f"{loss_prefix}{postfix}"
            avg_loss = torch.stack([x[loss_name] for x in outputs]).mean().double().detach().item()
            self.log(loss_name, avg_loss, sync_dist=True)

        log_loss("loss")
        log_loss("aux_loss")
        log_loss("main_loss")

    def training_step(self, batch: Any, batch_idx: int):
        return self.step(batch, self.train_metrics)

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self._epoch_end(outputs, self.train_metrics)

    def validation_step(self, batch: Any, batch_idx: int):
        return self.step(batch, self.valid_metrics, prefix="val/")

    def validation_epoch_end(self, outputs: List[Any]):
        self._epoch_end(outputs, self.valid_metrics, loss_prefix="val/")

    def test_step(self, batch: Any, batch_idx: int):
        return self.step(batch, self.test_metrics, prefix="test/")

    def test_epoch_end(self, outputs: List[Any]):
        self._epoch_end(outputs, self.test_metrics, loss_prefix="test/")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self._optimizer_callable(
            self.parameters(),
        )
        res = {"optimizer": optimizer}

        if self._scheduler_callable:
            scheduler_dict = {
                "scheduler": self._scheduler_callable(optimizer),
                "interval": "epoch",
            }
            res["lr_scheduler"] = scheduler_dict
        return res
