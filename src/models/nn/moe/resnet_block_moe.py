import copy
from functools import partial
from typing import List, Tuple, Type, Union

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

import src.models.nn.resnet
from src.models.nn import inject
from src.models.nn.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from src.models.nn.inject import apply_to_modules
from src.models.nn.moe.gate.topk import TopKGate
from src.models.nn.moe.layer import MOELayer, PatchMOELayer


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


def _randomize_weights(module: nn.Conv2d):
    """Add small amount of random noise to slightly change weight.

    Maintain same std deviation.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            with torch.no_grad():
                std = m.weight.std(dim=0)
                m.weight += 0.1 * std * torch.randn_like(m.weight)
                m.weight *= std / m.weight.std(dim=0)


def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, SynchronizedBatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class MoEModel(nn.Module, metaclass=PostInitCaller):
    def __init__(
        self,
        routing_layer_type: Type[nn.Module],
        num_experts: Union[List[int], int] = 16,
        noise_std: int = 0.1,
        expert_capacity=1.5,
        use_ste: bool = False,
        copy_weights: bool = False,
        balancing_loss: float = 0,
        k: int = 1,
        balancing_loss_type: str = "switch",
        variance_alpha: float = 0,
        **kwargs
    ):
        self.variance_alpha = variance_alpha
        self.balancing_loss_type = balancing_loss_type
        self.balancing_alpha = balancing_loss
        self.use_ste = use_ste
        self.expert_capacity = expert_capacity
        self.noise_std = noise_std
        self.num_experts = num_experts
        self.routing_layer_type = routing_layer_type
        self.copy_weights = copy_weights
        self.k = k

        assert (
            "moe_layer_type" not in kwargs
        ), "Overwrite replace() to specify a different MOELayer type."

        super().__init__(**kwargs)

    def __post_init__(self):
        # Inject moe layers into model
        if self.num_experts != 1:
            inject.apply_to_modules(self, self.replace, self.filter)

    def filter(self, module: nn.Module, prefix: str):
        raise NotImplementedError

    def replace(
        self,
        module: nn.Module,
        prefix: str,
        in_channels: int = None,
        moe_layer_type: Type[MOELayer] = MOELayer,
    ):
        if self.copy_weights:
            block_state_dict = module.state_dict()

            def _create_moe_layer():
                new_block = copy.deepcopy(module)
                new_block.load_state_dict(block_state_dict)
                _randomize_weights(new_block)
                return new_block

        else:

            def _create_moe_layer():
                m = copy.deepcopy(module)
                _initialize_weights(m)
                return m

        experts = nn.ModuleList([_create_moe_layer() for _ in range(self.num_experts)])

        routing_network = self.routing_layer_type(
            in_channels or module.in_channels, self.num_experts
        )

        gate = TopKGate(
            network=routing_network,
            epsilon=self.noise_std,
            expert_capacity=self.expert_capacity,
            use_straight_through_estimator=self.use_ste,
            balancing_alpha=self.balancing_alpha,
            k=self.k,
            balancing_loss_type=self.balancing_loss_type,
        )

        return moe_layer_type(gate, experts, var_alpha=self.variance_alpha)

    def reduce_to_fixed_expert(self, module_name: str, expert: int):
        """Replace moe layer with the chosen expert."""
        module: MOELayer = self.get_submodule(module_name)
        fixed_expert = module.experts[expert]

        def filter(_, prefix: str):
            print(
                module_name.strip("./"),
                prefix.strip("./"),
                module_name.strip("./") == prefix.strip("./"),
            )
            return module_name.strip("./") == prefix.strip("./")

        def new_module(*args):
            return fixed_expert

        apply_to_modules(self, new_module, filter)



class ResNetBlockMoE(MoEModel, src.models.nn.resnet.ResNet):
    def filter(self, module, prefix):
        return isinstance(module, (BasicBlock, Bottleneck)) and "layer4" in prefix

    def replace(self, module: nn.Module, prefix: str, in_channels: int = None):
        return super().replace(module, prefix, in_channels=module.conv1.in_channels)


class ResNetResidualBlockMoE(MoEModel, src.models.nn.resnet.ResNet):
    def filter(self, module, prefix):
        return isinstance(module, (BasicBlock, Bottleneck)) and "layer4.1" in prefix

    def replace(self, module: nn.Module, prefix: str, in_channels: int = None):
        return super().replace(module, prefix, in_channels=module.conv1.in_channels)

