import pytest
import segmentation_models_pytorch
import torch
import torchattacks
from torchvision.models import resnet18
from torchvision.transforms import transforms

from src.models.nn.attacks import AttackModule


@pytest.mark.parametrize(
    "std_mean",
    [
        (
            (
                0.5,
                0.4,
                0.3,
            ),
            (0.1, -0.1, 0.4),
        )
    ],
)
@pytest.mark.parametrize("attack_class", [torchattacks.GN, torchattacks.PGD])
@pytest.mark.parametrize(
    "model",
    [
        resnet18(pretrained=False, num_classes=20),
        segmentation_models_pytorch.FPN("resnet18", in_channels=3, classes=20),
    ],
)
def test_apply_attack(std_mean, attack_class, model):
    std, mean = std_mean
    num_classes = 20

    model = resnet18(pretrained=False, num_classes=num_classes)
    normalization = transforms.Normalize(mean, std)
    attack_module = AttackModule(model, normalization, attack_class)

    x = torch.randn((2, len(attack_module.normalization.mean), 28, 28))
    if isinstance(model, segmentation_models_pytorch.FPN):
        y = torch.randint(low=0, high=num_classes - 1, size=(2, 28, 28))
    else:
        y = torch.randint(low=0, high=num_classes - 1, size=(2,))

    x_pert, y_pert = attack_module((x, y))
    assert x_pert.shape == x.shape
    assert (y - y_pert).abs().sum() == 0
