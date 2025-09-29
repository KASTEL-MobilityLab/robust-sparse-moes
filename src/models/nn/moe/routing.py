import torch
from torch import nn


class GlobalAvgLinearRoutingNetwork(nn.Sequential):
    def __init__(self, inplanes, num_experts):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(inplanes, num_experts),
            nn.Softmax(),
        )


class ConvGlobalAvgRoutingNetwork(nn.Sequential):
    def __init__(self, inplanes, num_experts, kernel_size=(1, 1), bias=False):
        super().__init__(
            nn.Conv2d(inplanes, num_experts, kernel_size=kernel_size, bias=bias),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Softmax(),
        )

class DoubleConvGlobalAvgRoutingNetwork(nn.Sequential):
    def __init__(self, inplanes, num_experts, midplanes=8, kernel_size=(3, 3), bias=False):
        super().__init__(
            nn.Conv2d(inplanes, midplanes,
                      kernel_size=kernel_size,
                      bias=bias),
            nn.BatchNorm2d(midplanes),
            nn.Conv2d(midplanes, num_experts,
                      kernel_size=kernel_size,
                      bias=bias),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Softmax(),
        )


class GlobalAvgDeterministicRoutingNetwork(nn.Sequential):
    def __init__(self, inplanes, num_experts):
        super().__init__()
        self.register_buffer("support_vectors", torch.randn((num_experts, inplanes)))

        self.pre_processing = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.post_processing = nn.Softmax()

    def forward(self, input):
        processed = self.pre_processing(input)

        # processed.shape = (tokens, inplanes)
        # Deterministically project onto vectors
        similarities = processed @ self.support_vectors.T

        propabilities = self.post_processing(similarities)

        # propabilities.shape (tokens, num_experts)
        return propabilities


class SingleEntryDeterministicRoutingNetwork(nn.Sequential):
    def __init__(self, inplanes, num_experts):
        super().__init__()
        self.num_experts = num_experts

        self.pre_processing = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.post_processing = nn.Softmax()

    def forward(self, input):
        processed = self.pre_processing(input)

        # processed.shape = (tokens, inplanes)
        # Just select first num_expert entries:
        similarities = processed[:, : self.num_experts]

        propabilities = self.post_processing(similarities)

        # probabilities.shape (tokens, num_experts)
        return propabilities
