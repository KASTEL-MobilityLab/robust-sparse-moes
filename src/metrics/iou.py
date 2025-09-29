import torch
import torchmetrics
from torch import Tensor


class IoU(torchmetrics.JaccardIndex):
    """IoU implementation that treats ignore_class the same as cross entropy loss.

    Just don't look at the following code.

    It's a complete mess. For some reason torchmetrics does not deal with ignore_index in the same way that torch does
    it. Resulting in the annoyance that either ignore_index is out of bounce or the prediction tensor
    has to include the ignored class.

    "Solution":

    Basically just take any ignore_index value and change it to num_classes. Increase num_classes by one.
    And at runtime adjust target's ignore_index entries to num_classes. Then attach one empty class to
    predictions.
    """

    def __init__(self, task, num_classes, ignore_index=None, *args, **kwargs):
        self.old_ignore_index = ignore_index
        if ignore_index:
            ignore_index = num_classes
            num_classes += 1
        super().__init__(task=task, num_classes=num_classes, ignore_index=ignore_index, *args, **kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.old_ignore_index:
            target[target == self.old_ignore_index] = self.ignore_index

        super().update(preds, target)
