from pydoc import locate
from typing import Any, OrderedDict, Type

import pytorch_lightning
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import transforms


def make_lit_module(base_class: str, *args, **kwargs):
    """Create an instance of subclass that is attacked by a given attack."""
    # resolve class
    base_class: Type[pytorch_lightning.LightningModule] = locate(base_class)

    # Create class
    class AdvTrainFastLitModule(base_class):
        """LightningModule for fast adversarial training classification."""

        def __init__(
            self,
            normalization: transforms.Normalize,
            ignore_index: int = 255,
            clip_eps=4 / 255,
            fgsm_step=4 / 255,
            *args,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.ignore_index = ignore_index
            self.clip_eps = clip_eps
            self.fgsm_step = fgsm_step
            self.normalization = normalization
            self.inverse_normalization = transforms.Compose(
                [
                    transforms.Normalize(mean=[0.0], std=[1 / x for x in normalization.std]),
                    transforms.Normalize(mean=[-x for x in normalization.mean], std=[1.0]),
                ]
            )
            self.attack_active = True

        def _fast_adv_step(self, input, target):
            input = self.inverse_normalization(input)

            uniform = Variable(
                2 * self.clip_eps * (torch.rand_like(input) - 0.5), requires_grad=True
            )

            loss = F.cross_entropy(
                self.model(self.normalization(input + uniform)),
                target.long(),
                ignore_index=self.ignore_index,
            )
            loss.backward()

            noise = self.fgsm_step * uniform.grad.detach().sign()
            noise.clamp_(-self.clip_eps, self.clip_eps)

            input = input + noise
            input.clamp_(0, 1)

            return self.normalization(input).detach()

        def partial_step(self, batch: Any):
            if self.attack_active and self.training:
                input, target = batch
                batch = self._fast_adv_step(input, target), target

            return super().partial_step(batch)

        def load_state_dict(self, state_dict: "OrderedDict[str, Tensor]", strict: bool = False):
            return super().load_state_dict(state_dict, strict=strict)

    # Return instance
    return AdvTrainFastLitModule(*args, **kwargs)
