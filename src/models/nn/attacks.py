import math
from typing import Any, Callable, Tuple, Union

import torch
import torchattacks
from torchvision.transforms import transforms


def _prepare_output(y: Union[torch.Tensor, tuple]) -> torch.Tensor:
    """Reshape any input into a (batch,classes) shape."""
    if isinstance(y, tuple):
        y = y[0]
    assert isinstance(
        y, torch.Tensor
    ), "Model output has to be either a tensor or a tuple with a tensor as it's first output."

    if y.dtype in {torch.uint8, torch.int, torch.long}:
        # Structure is then (batch, other_dims)
        return y.clone().reshape(math.prod(y.shape))
    else:
        batch_size, num_classes, *other_dims = y.shape
        y = y.clone().reshape(batch_size * math.prod(other_dims), num_classes)
        return y


class _FlattenModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        model_out = self.model(x)
        flattened = _prepare_output(model_out)
        return flattened


class AttackModule(torch.nn.Module):
    """Module for attacks."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalization: transforms.Normalize,
        attack: Callable,
        ignore_index: int = 255,
        flatten_inputs: bool = True,
    ):
        super().__init__()
        self.normalization = normalization
        self.inverse_normalization = transforms.Compose(
            [
                transforms.Normalize(mean=[0.0], std=[1 / x for x in normalization.std]),
                transforms.Normalize(mean=[-x for x in normalization.mean], std=[1.0]),
            ]
        )
        self.normalization.requires_grad_(False)
        self.inverse_normalization.transforms[0].requires_grad_(False)
        self.inverse_normalization.transforms[1].requires_grad_(False)

        self.attack = attack
        self.ignore_index = ignore_index

    def forward(self, model, batch: Tuple[torch.Tensor, torch.Tensor, Any]):
        x, y, *others = batch

        # TODO: Not a nice solution
        pert_x = self.inverse_normalization(x.clone().detach())
        pert_x = pert_x.to(x.device)

        pert_x, _ = self.attack(
            model,
            normalization=self.normalization,
            X=pert_x,
            y=y.long(),
            ignore_index=self.ignore_index,
        )

        pert_x = self.normalization(pert_x)
        pert_x = pert_x.clone().detach().to(x.device)

        batch = pert_x, y
        return *batch, *others
