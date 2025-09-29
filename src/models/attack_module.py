from pydoc import locate
from typing import Any, Callable, List, OrderedDict

from torch import Tensor
from torchvision.transforms import transforms

from src.models.nn.attacks import AttackModule


class _AttackLitModule:
    """Dummy class to use for isinstance() checks."""

    pass


def make_lit_module(base_class: str, flatten_inputs: bool = True, *args, **kwargs):
    from pydoc import locate
    base_class = locate(base_class)
    attack_fn = kwargs.pop("attack", None)
    test_metric_prefix = kwargs.pop("test_metric_prefix", "attack/")

    class AttackLitModule(base_class, _AttackLitModule):
        def __init__(
            self,
            normalization: transforms.Normalize,
            ignore_index: int = 255,
            *args,
            **kwargs
        ):
            super().__init__(*args, test_metric_prefix=test_metric_prefix, **kwargs)
            self.attack_module = AttackModule(
                self.model, normalization, attack_fn, ignore_index, flatten_inputs=flatten_inputs
            )
            self.attack_active = True

        def partial_step(self, batch: Any):
            if self.attack_active:
                batch = self.attack_module(self.model, batch)
            return super().partial_step(batch)

        def load_state_dict(self, state_dict: "OrderedDict[str, Tensor]", strict: bool = False):
            return super().load_state_dict(state_dict, strict=strict)

        def test_step(self, batch: Any, batch_idx: int):
            return self.step(batch, self.test_metrics, prefix=test_metric_prefix)

        def test_epoch_end(self, outputs: List[Any]):
            self._epoch_end(outputs, self.test_metrics, loss_prefix=test_metric_prefix)

    return AttackLitModule(*args, **kwargs)

