import copy
from typing import List, Type, Union

from torch import nn
from torch.nn import Conv2d

import src.models.nn.resnet
from src.models.nn.moe.layer import MOELayer
from src.models.nn.moe.resnet_block_moe import MoEModel


class ResNetConvMoE(MoEModel, src.models.nn.resnet.ResNet):
    def __init__(self, moe_layer_prefix: str = "layer4", **kwargs):
        self.moe_layer_prefix = moe_layer_prefix

        super().__init__(**kwargs)

    def filter(self, module: nn.Module, prefix: str):
        return isinstance(module, Conv2d) and self.moe_layer_prefix in prefix
