from collections import Iterable
from pydoc import locate
from typing import OrderedDict, Type

import pytorch_lightning
import torch
from torch import Tensor
from torchvision.transforms import transforms


def make_lit_module(base_class: str, *args, **kwargs):
    """Create an instance of subclass that is attacked by a given attack."""
    # resolve class
    base_class: Type[pytorch_lightning.LightningModule] = locate(base_class)

    # Create class
    class AdvTrainLitModule(base_class):
        """LightningModule for free adversarial training classification."""

        def __init__(
            self,
            normalization: transforms.Normalize,
            clip_eps=4 / 255,
            fgsm_step=4 / 255,
            n_repeats=4,
            p_reset_noise: float = 0,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.automatic_optimization = False

            self.normalization = normalization
            self.inverse_normalization = transforms.Compose(
                [
                    transforms.Normalize(mean=[0.0], std=[1 / x for x in normalization.std]),
                    transforms.Normalize(mean=[-x for x in normalization.mean], std=[1.0]),
                ]
            )

            self.global_noise_data = None
            self.n_repeats = n_repeats
            self.clip_eps = clip_eps
            self.fgsm_step = fgsm_step
            self.p_reset_noise = p_reset_noise

        def training_step(self, batch, batch_idx):
            x, y = batch

            if self.global_noise_data is None:
                self.global_noise_data = torch.zeros_like(x)
            elif self.global_noise_data.shape[0] < x.shape[0]:
                self.logger.warning(
                    f"Unexpect input shape:{x.shape} with global noise shape: {self.global_noise_data.shape}"
                )
                self.global_noise_data = torch.zeros_like(x)

            total_loss = 0
            total_aux_loss = 0

            total_preds = []
            total_targets = []
            for _ in range(self.n_repeats):

                # Reset part of the noise to 0!
                if self.p_reset_noise > 0:
                    discard = torch.rand(x.shape[0]) <= self.p_reset_noise
                    self.global_noise_data[discard] = torch.zeros_like(x)[discard]

                # Ascend on the global noise
                noise_batch = self.global_noise_data[: x.shape[0]]
                noise_batch.requires_grad = True

                assert (
                    noise_batch.shape == x.shape
                ), f"Noise shape mismatch: {noise_batch.shape} != {x.shape}"

                in1 = self.inverse_normalization(x) + noise_batch
                in1.clamp_(0, 1.0)
                in1 = self.normalization(in1)

                # compute gradient and do SGD partial_step
                for optim in self._iterable_optims():
                    optim.zero_grad()

                _, preds, _, loss, aux_loss = self.partial_step((in1, y))
                total_preds.append(preds)
                total_targets.append(y)

                loss = loss + aux_loss
                self.manual_backward(loss + aux_loss)

                total_loss += loss.detach().cpu()
                total_aux_loss += aux_loss.detach().cpu()

                # Update the noise for the next iteration
                pert = self.fgsm_step * torch.sign(noise_batch.grad)
                self.global_noise_data[: x.shape[0]] += pert.data
                self.global_noise_data.clamp_(-self.clip_eps, self.clip_eps)

                for optim in self._iterable_optims():
                    optim.step()

            loss = total_loss / self.n_repeats
            aux_loss = total_aux_loss / self.n_repeats

            total_targets = torch.cat(total_targets)
            total_preds = torch.cat(total_preds)

            self.train_metrics.update(total_preds, total_targets)

            output = {
                "loss": loss + aux_loss,
                "main_loss": loss,
                "aux_loss": aux_loss,
            }
            if self.return_results_on_step:
                output.update(
                    {
                        "preds": total_preds.detach().cpu(),
                        "targets": total_targets.detach().cpu(),
                    }
                )

            # log train metrics
            return output

        def _iterable_schedulers(self):
            scheds = self.lr_schedulers()
            return scheds if isinstance(scheds, Iterable) else [scheds]

        def _iterable_optims(self):
            optims = self.optimizers()
            return optims if isinstance(optims, Iterable) else [optims]

        def on_train_epoch_end(self) -> None:
            for scheduler in self._iterable_schedulers():
                scheduler.step()

        def load_state_dict(self, state_dict: "OrderedDict[str, Tensor]", strict: bool = False):
            return super().load_state_dict(state_dict, strict=strict)

    # Return instance
    return AdvTrainLitModule(*args, **kwargs)
