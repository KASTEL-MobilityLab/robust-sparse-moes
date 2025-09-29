from typing import Tuple

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import transforms
from torchattacks import APGD

def fgsm(
    model, X: torch.Tensor, y: torch.Tensor, eps: float, ignore_index: int = 255
) -> torch.Tensor:
    """Construct FGSM adversarial examples on the examples X."""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = F.cross_entropy(model(X + delta), y, ignore_index=ignore_index)
    loss.backward()
    return (X + eps * delta.grad.detach().sign()).detach()


def pgd(
    model: torch.nn.Module,
    normalization: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    ignore_index: int = 255,
    random_start: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct PGD adversarial examples on the examples X."""
    X_pert = Variable(X, requires_grad=True)
    if random_start:
        X_pert.data = X_pert + 2 * eps * (torch.rand_like(X_pert) - 0.5)
        X_pert.data = X_pert.clamp(0, 1)

    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    with torch.enable_grad():
        for t in range(steps):
            loss = ce(model(normalization(X_pert)), y.clone())
            loss.backward()
            X_pert.data = X_pert + alpha * torch.sign(X_pert.grad)
            X_pert.data = X_pert.clamp(X - eps, X + eps).clamp(0, 1)
            X_pert.grad.zero_()

    return X_pert.detach(), None


def auto_pgd(
    model,
    normalization,
    X,
    y,
    eps: float = 8/255,
    steps: int = 20,
    ignore_index: int = 255,
    **kwargs,
):
    class WrappedModel(torch.nn.Module):
        def __init__(self, model, normalization):
            super().__init__()
            self.model = model
            self.norm = normalization

        def forward(self, x):
            return self.model(self.norm(x))

    wrapped_model = WrappedModel(model, normalization)
    attack = APGD(wrapped_model, eps=eps, steps=steps)
    # attack.set_return_type("float")

    adv_images = attack(X, y)
    return adv_images.detach(), None


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10)
    )
    normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    X = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))

    adv_X, _ = auto_pgd(model, normalization, X, y)
    print(adv_X.shape)
