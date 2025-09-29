import torchvision
from torch import nn
from torch.nn import Conv2d


class ResNet(nn.Module):
    def __init__(self, layers: int, num_channels: int = 1, small_inputs: bool = False, **kwargs):
        super().__init__()
        model: torchvision.models.ResNet = self.get_model(layers, **kwargs)

        if small_inputs:
            model.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        elif num_channels != 3:
            model.conv1 = Conv2d(
                num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        self.model = model

    def get_model(self, layers: int, **kwargs):
        name = f"resnet{layers}"

        return getattr(torchvision.models, name)(**kwargs)

    def forward(self, x):
        return self.model(x)
