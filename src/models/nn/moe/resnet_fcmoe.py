import torchvision
from torch import nn
from torch.nn import ModuleList

import src.models.nn.resnet
from src.models.nn.moe.gate.top1 import Top1Gate
from src.models.nn.moe.layer import SkipMOELayer


class ResNetFCMoE(src.models.nn.resnet.ResNet):
    def __init__(
        self,
        num_experts: int = 16,
        noise_std: int = 0.1,
        expert_capacity=1.5,
        use_ste: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features

        experts = ModuleList([nn.Linear(in_features, out_features) for _ in range(num_experts)])

        routing_network = nn.Sequential(nn.Linear(in_features, num_experts), nn.Softmax())

        gate = Top1Gate(
            network=routing_network,
            epsilon=noise_std,
            expert_capacity=expert_capacity,
            use_straight_through_estimator=use_ste,
        )

        self.model.fc = SkipMOELayer(gate, experts, output_token_shape=(out_features,))

    def forward(self, x):
        return self.model(x)
