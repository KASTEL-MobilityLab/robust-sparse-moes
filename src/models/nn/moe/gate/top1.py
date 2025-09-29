"""Copyright 2021 The Microsoft DeepSpeed Team."""
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.nn.moe.gate.simple_gate import AbstractGate

logger = logging.getLogger(__name__)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    capacity = torch.maximum(capacity, min_capacity).to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=int(k.item()), dim=0)[1]


@torch.jit.script
def _top_idx_list(source, k_list: torch.Tensor):
    idx_list = []

    for d, k in enumerate(k_list):
        idx_list.append(torch.topk(source[:, d], k=k)[1])

    return idx_list


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def top1gating(
    scores: Tensor,
    capacity: Union[int, torch.Tensor],
) -> Tuple[Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    num_tokens, num_experts = scores.shape

    # everything is in fp32 in this function
    gates = F.softmax(scores, dim=1)

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(gates * capacity.view(1, -1), dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    if capacity.min() < num_tokens:
        if len(capacity.shape) > 0:
            # Annoyingly complex solution because top_indices is a list of tensors of different size
            top_indices = _top_idx_list(scores, capacity)
            new_mask1 = torch.zeros_like(mask1)
            for i, top_idx in enumerate(top_indices):
                new_mask1[:, i] = mask1[:, i] * torch.zeros_like(mask1[:, i]).scatter_(
                    0, top_idx, 1
                )
        else:
            top_idx = _top_idx(mask1, capacity)
            new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
        mask1 = new_mask1

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach()

    return mask1, exp_counts


class Top1Gate(AbstractGate):
    def __init__(
        self,
        network: torch.nn.Module,
        epsilon: float = 0.1,
        expert_capacity: float = float("inf"),
        use_straight_through_estimator: bool = False,
        max_iter: int = 25,
        balancing_alpha: float = 0,
    ):
        """General Top1 gate that learns to calculate a routing mask for inputs.

        Training:

        The Top1 calculation is done greedily by routing each input to the expert with maximum score.
        Once an expert's capacity is reached, it's respective scores are redacted. In another iteration the inputs are
        then assigned to the next best expert.
        This is done until all inputs are distributed.

        Evaluation:

        The Top1 calculation is done without respecting the capacity by routing each input to the expert with maximum
        score.


        Args:
            network: Calculate score matrix inputs x experts
            epsilon: Noise added to scores during training.
            expert_capacity: Relative capacity of each expert.
            use_straight_through_estimator: Divide score values by itself (detached). This way gradients are
            propagated but the score values are ignored.
            max_iter: Max number of attempts to balance inputs to experts.
        """
        super().__init__()
        self.balancing_alpha = balancing_alpha
        self.use_straight_through_estimator = use_straight_through_estimator
        self.expert_capacity = expert_capacity
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.k = 1

        self.network = network

    def _top1_with_reassignment(
        self, scores: torch.Tensor, high_negative_score: float = -1e5
    ) -> torch.Tensor:
        """Distribute scores with greedy reassignment to experts."""
        batch_size, num_experts = scores.shape
        selected_tokens = set()

        capacity = _capacity(
            scores, torch.tensor(min(self.expert_capacity, batch_size)), torch.tensor(1)
        )
        capacity = capacity * torch.ones((num_experts,), device=scores.device, dtype=torch.int64)

        mask = torch.zeros_like(scores)

        for i in range(self.max_iter):

            current_capacity = capacity
            if len(capacity.unique()) == 1:
                current_capacity = current_capacity[0]

            if i == self.max_iter - 1:
                # Distribute all missing inputs in last iteration
                current_capacity = torch.tensor(batch_size - len(selected_tokens))

            add_mask, exp_counts = top1gating(
                scores,
                capacity=current_capacity,
            )

            selected_tokens = selected_tokens.union(
                set(x.to(torch.int64).item() for x in torch.where(add_mask.sum(dim=1) > 0)[0])
            )
            scores[tuple(selected_tokens), :] = high_negative_score
            capacity -= exp_counts
            mask[mask.sum(dim=1) == 0] = (
                mask[mask.sum(dim=1) == 0] + add_mask[mask.sum(dim=1) == 0]
            )

            if len(selected_tokens) == batch_size:
                # logger.debug(f"Finished in iter {i}")
                break
            elif i == self.max_iter - 1:
                tokens_per_experts = (mask != 0).sum(0)
                logger.warning(tokens_per_experts)

        return mask

    @staticmethod
    def _top1_without_reassignment(scores: torch.Tensor) -> torch.Tensor:
        """Distribute scores without regarding capacity to experts."""
        batch_size = scores.shape[0]
        capacity = torch.tensor(batch_size, dtype=scores.dtype, device=scores.device)

        mask, _ = top1gating(
            scores,
            capacity=capacity,
        )

        return mask

    def _balancing_loss(self, scores: Tensor):
        """From switch transformer.

        T = num_tokens
        N = num_experts
        """
        T, N = scores.shape

        f = 1 / T * (scores == scores.max(dim=1, keepdim=True)[0])
        P = 1 / T * scores.sum(dim=0)

        return (f * P).sum()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scores = self.network(input)

        # Also calculate loss!
        self.loss = self.balancing_alpha * self._balancing_loss(scores)

        if self.training:
            # Add gaussian noise
            scores_ = scores / scores.std()
            scores_ = scores_ + torch.normal(
                0, std=self.epsilon, size=scores.shape, device=scores.device
            )

            mask = self._top1_with_reassignment(scores_)
        else:
            # During evaluation, the tokens should be routed to the best fitting expert!
            mask = self._top1_without_reassignment(scores)

        # Gradient is only propagated along scores!
        if self.use_straight_through_estimator:
            return STEFunction.apply(scores * mask.detach())
        else:
            return scores * mask.detach()
