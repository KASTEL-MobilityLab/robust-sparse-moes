import math
from typing import Type

import pytest
import torch
from torch.nn import ModuleList

from src.models.nn.moe.gate.simple_gate import RandomTopKGate, SimpleTopKGate
from src.models.nn.moe.gate.top1 import Top1Gate
from src.models.nn.moe.gate.topk import TopKGate
from src.models.nn.moe.layer import MOELayer, PatchMOELayer


@pytest.mark.parametrize("gate_type", [RandomTopKGate, SimpleTopKGate])
@pytest.mark.parametrize("output_token_shape", [(8,), (10,)])
def test_simple_moe_layer(
    gate_type: Type[SimpleTopKGate], output_token_shape: tuple, k=1, num_experts=4
):
    in_features = 5
    input_dims = [(1, in_features), (10, in_features)]
    output_dims = [(1, *output_token_shape), (10, *output_token_shape)]

    gate = gate_type(num_experts=num_experts, k=k, in_features=in_features)
    experts = ModuleList([torch.nn.Linear(5, *output_token_shape) for _ in range(num_experts)])
    moe_layer = MOELayer(gate, experts)

    for in_dim, out_dim in zip(input_dims, output_dims):
        input = torch.randn(in_dim)
        output = moe_layer(input)
        assert output.shape == out_dim


@pytest.mark.parametrize("patch_size", [(2, 2), (12, 12)])
def test_simple_moe_layer(patch_size: tuple, k=1, num_experts=4):
    in_features = 1
    input_dims = [
        (1, in_features, 2 * patch_size[0], 1 * patch_size[1]),
        (10, in_features, 4 * patch_size[0], 3 * patch_size[1]),
    ]

    gate = RandomTopKGate(num_experts=num_experts, k=k, in_features=in_features)
    experts = ModuleList([torch.nn.Identity() for _ in range(num_experts)])
    moe_layer = PatchMOELayer(patch_size, gate=gate, experts=experts)

    for in_dim in input_dims:
        input = torch.randn(in_dim)
        output = moe_layer(input)
        print(((input - output).abs() < 1e-1))
        assert ((input - output).abs() < 1e-1).all()


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("expert_capacity", [float("inf"), 1, 1.5, 2])
def test_simple_gate(k, expert_capacity, num_experts=32, in_features=16):
    gate = SimpleTopKGate(num_experts, in_features, k=k, expert_capacity=expert_capacity)

    input = torch.randn((2, in_features))

    output = gate(input)
    assert torch.count_nonzero(output[0]) == k


@pytest.mark.parametrize("expert_capacity", [float("inf"), 1.5, 2, 16])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 16])
@pytest.mark.parametrize("training", [True])
def test_top1_gate(expert_capacity, batch_size, training, num_experts=4, in_features=2):
    network = torch.nn.Linear(in_features, num_experts)

    gate = Top1Gate(network=network, expert_capacity=expert_capacity)

    gate.train(training)

    input = torch.randn((batch_size, in_features))

    output = gate(input)
    for o in output:
        assert torch.count_nonzero(o) == 1


@pytest.mark.parametrize("expert_capacity", [float("inf"), 1.5, 2, 16])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 16])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("use_ste", [True, False])
@pytest.mark.parametrize("normalize_routing", [True, False])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_topk_gate(
    expert_capacity,
    batch_size,
    training,
    use_ste,
    normalize_routing,
    k,
    num_experts=4,
    in_features=2,
):
    if k > 1:
        use_ste = False
    network = torch.nn.Linear(in_features, num_experts)

    gate = TopKGate(
        network=network,
        expert_capacity=expert_capacity,
        k=k,
        max_iter=25,
        use_straight_through_estimator=use_ste,
        normalize_routing=normalize_routing,
    )

    gate.train(training)

    input = torch.randn((batch_size, in_features))

    output = gate(input)

    assert torch.all(
        torch.count_nonzero(output, dim=1) == k
    ), f"Some tokens are not distributed to k experts: {output}"
    if expert_capacity != float("inf") and training:
        assert torch.all(
            torch.count_nonzero(output, dim=0)
            <= math.ceil(k * batch_size * expert_capacity / num_experts)
        ), f"Capacity of some experts is not respected: {output}"
