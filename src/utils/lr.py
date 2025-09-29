from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    _LRScheduler,
)


class _LinearWarmupLR(ChainedScheduler):
    def __init__(self, main_scheduler: _LRScheduler, start_factor=0.01, warmup_epochs: int = 0):
        self.optimizer = main_scheduler.optimizer

        warm_up = LinearLR(
            self.optimizer, start_factor=start_factor, end_factor=1, total_iters=warmup_epochs
        )
        super().__init__(schedulers=(warm_up, main_scheduler))


class PolyLR(_LinearWarmupLR):
    """Polynomial learning rate scheduler with a linear warmup."""

    def decay(self, epoch):
        return (1 - epoch / self.T_max) ** self.exponent

    def __init__(self, optimizer, T_max: int, exponent: float = 0.9, *args, **kwargs):
        self.T_max = T_max
        self.exponent = exponent

        poly_lr = LambdaLR(optimizer, lr_lambda=self.decay)

        super().__init__(poly_lr, *args, **kwargs)


class CosineLR(_LinearWarmupLR):
    """Cosine learning rate scheduler with a linear warmup."""

    def __init__(self, optimizer, T_max, warmup_epochs: int, *args, **kwargs):
        cosine = CosineAnnealingLR(optimizer, T_max=T_max - warmup_epochs, *args, **kwargs)

        super().__init__(cosine, warmup_epochs=warmup_epochs)
