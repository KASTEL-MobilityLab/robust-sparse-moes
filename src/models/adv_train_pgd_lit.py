from collections.abc import Iterable
from pydoc import locate
from typing import Any, Callable, OrderedDict, Type

import pytorch_lightning
import torch
from torch import Tensor
from torchvision.transforms import transforms

from src.models.nn.attacks import AttackModule


def make_lit_module(base_class: str, *args, **kwargs):
    """Create an instance of subclass that is attacked by a given attack."""
    # resolve class
    base_class: Type[pytorch_lightning.LightningModule] = locate(base_class)

    # Create class
    class AdvTrainPGDLitModule(base_class):
        """LightningModule for pgd adversarial training classification."""

        def __init__(
            self,
            normalization: transforms.Normalize,
            attack: Callable,
            ignore_index: int = 255,
            *args,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.attack_module = AttackModule(
                self.model, normalization, attack, ignore_index, flatten_inputs=False
            )
            self.attack_active = True

        def partial_step(self, batch: Any):
            if self.attack_active and self.training:
                self.attack_module.eval()
                batch = self.attack_module(self.model, batch)
            return super().partial_step(batch)

        def load_state_dict(self, state_dict: "OrderedDict[str, Tensor]", strict: bool = False):
            return super().load_state_dict(state_dict, strict=strict)

    # Return instance
    return AdvTrainPGDLitModule(*args, **kwargs)
