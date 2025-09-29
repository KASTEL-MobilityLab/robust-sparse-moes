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
from torch import Tensor, nn

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
        return (input != 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor, k: int = 1) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil(k * (num_tokens / num_experts) * capacity_factor).to(torch.int64)
    capacity = torch.maximum(capacity, min_capacity).to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=min(int(k.item()), source.shape[0]), dim=0)[1]


@torch.jit.script
def _top_idx_list(source, k_list: torch.Tensor):
    idx_list = []

    for d, k in enumerate(k_list):
        idx_list.append(torch.topk(source[:, d], k=k)[1])

    return idx_list


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def _enforce_capacity(scores, mask, capacity, dim):
    assert dim in {0, 1}, f"Function is only defined for dim 0 or 1, not {dim}."

    if dim == 0:
        mask = mask.transpose(0, 1)
        scores = scores.transpose(0, 1)

    capacity[capacity > scores.shape[0]] = scores.shape[0]

    # Annoyingly complex solution because top_indices is a list of tensors of different size
    top_indices = _top_idx_list(scores * mask, capacity)
    new_mask1 = torch.zeros_like(mask)
    for i, top_idx in enumerate(top_indices):
        new_mask1[:, i] = mask[:, i] * torch.zeros_like(mask[:, i]).scatter_(0, top_idx, 1)

    mask = new_mask1

    if dim == 0:
        mask = mask.transpose(0, 1)
        scores = scores.transpose(0, 1)

    return mask


def topkgating(
    scores: Tensor, capacity: Union[int, torch.Tensor], ks: torch.Tensor
) -> Tuple[Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(scores, dim=1)

    mask1 = torch.outer(ks, capacity)
    mask1[mask1 != 0] = 1

    # Enforce max k per token
    mask1 = _enforce_capacity(gates, mask1, ks, dim=0)

    # Enforce expert capacity
    mask1 = _enforce_capacity(gates, mask1, capacity, dim=1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().long()

    return mask1, exp_counts


class TopKGate(AbstractGate):
    def __init__(
        self,
        network: torch.nn.Module,
        epsilon: float = 0.1,
        expert_capacity: float = float("inf"),
        use_straight_through_estimator: bool = False,
        max_iter: int = 25,
        balancing_alpha: float = 0,
        k: int = 1,
        normalize_routing: bool = True,
        balancing_loss_type: str = "switch",
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
        self.normalize_routing = normalize_routing
        self.balancing_alpha = balancing_alpha
        self.use_straight_through_estimator = use_straight_through_estimator
        self.expert_capacity = expert_capacity or float("inf")
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.k = k
        self.balancing_loss_type = balancing_loss_type

        if self.k != 1:
            self.use_straight_through_estimator = False
            if use_straight_through_estimator:
                logger.warning(
                    "STE should only be used with k=1. It's been automatically turned off."
                )

        self.network = network

    def _topk_with_reassignment(
        self, scores: torch.Tensor, high_negative_score: float = -1e5
    ) -> torch.Tensor:
        """Distribute scores with greedy reassignment to experts."""
        batch_size, num_experts = scores.shape

        capacity = _capacity(
            scores, torch.tensor(min(self.expert_capacity, batch_size)), torch.tensor(1), k=self.k
        )
        capacity = capacity * torch.ones((num_experts,), device=scores.device, dtype=torch.int64)

        mask = torch.zeros_like(scores)

        for i in range(self.max_iter):

            current_capacity = capacity

            if i == self.max_iter - 1:
                # Distribute all missing inputs in last iteration
                current_capacity = batch_size * torch.ones(
                    (num_experts,), device=scores.device, dtype=torch.int64
                )

            add_mask, exp_counts = topkgating(
                scores,
                capacity=current_capacity,
                ks=self.k - mask.sum(dim=1).long(),
            )

            mask = mask + add_mask

            scores[mask != 0] = high_negative_score
            capacity -= exp_counts

            if (mask != 0).sum() == self.k * batch_size:
                # logger.debug(f"Finished in iter {i}")
                break
            elif i == self.max_iter - 1:
                tokens_per_experts = (mask != 0).sum(0)
                logger.warning(tokens_per_experts)

        return mask

    def _topk_without_reassignment(self, scores: torch.Tensor) -> torch.Tensor:
        """Distribute scores without regarding capacity to experts."""
        batch_size, num_experts = scores.shape
        capacity = batch_size * torch.ones((num_experts,), device=scores.device, dtype=torch.int64)
        ks = self.k * torch.ones((batch_size,), device=scores.device, dtype=torch.int64)

        mask, _ = topkgating(scores, capacity=capacity, ks=ks)

        return mask

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        scores = self.network(input)

        # Also calculate loss!
        self.loss = self.balancing_alpha * _balancing_loss(scores, self.balancing_loss_type)

        if self.training and self.balancing_loss_type == "entropy":
            mask = self._topk_without_reassignment(scores)
            # Otherwise the following is wrong!
            assert (
                self.expert_capacity == float("inf") or self.expert_capacity is None
            ), self.expert_capacity
        elif self.training:
            # Add gaussian noise
            scores_ = scores / scores.std()
            scores_ = scores_ + torch.normal(
                0, std=self.epsilon, size=scores.shape, device=scores.device
            )

            mask = self._topk_with_reassignment(scores_)
        else:
            # During evaluation, the tokens should be routed to the best fitting expert!
            mask = self._topk_without_reassignment(scores)

        result = None

        # Gradient is only propagated along scores!
        if self.use_straight_through_estimator:
            result = STEFunction.apply(scores * mask.detach())
        elif self.normalize_routing:
            out = scores * mask.detach()
            result = out / (out.sum(dim=1).view(-1, 1) + 1e-5)
        else:
            result = scores * mask.detach()

        #assert not result.isnan().any(), f"Calculated mask contains NAN: {result}"
        if result.isnan().any():
            result = torch.nan_to_num(result)

        return scores, result


@torch.jit.script
def _entropy(p: torch.Tensor):
    return -(p * (p + (p == 0) * 1e-5).log()).sum()


@torch.jit.script
def _balancing_loss(scores: Tensor, loss_type: str = "switch"):
    """From switch transformer.

    T = num_tokens
    N = num_experts
    """
    T, N = scores.shape

    if loss_type == "switch":
        mask = torch.zeros_like(scores).scatter_(1, scores.argmax(dim=1).unsqueeze(1), 1.0)

        f = 1 / T * mask.sum(dim=0).detach()

        P = 1 / T * scores.sum(dim=0)

        return N * (f * P).sum() - 1
    elif loss_type == "variance":
        mask = torch.zeros_like(scores).scatter_(1, scores.argmax(dim=1).unsqueeze(1), 1.0)

        f = 1 / T * mask.sum(dim=0).detach()

        P = 1 / T * scores.sum(dim=0)

        switch_loss = N * (f * P).sum() - 1
        return switch_loss - scores.std(dim=0).sum() / N + 1
    elif loss_type == "entropy":

        columns = 1 / T * scores.sum(dim=0)

        entropy_columns = _entropy(columns)
        entropy_tokens = _entropy(scores)

        return 1 / T * entropy_tokens - entropy_columns
    elif loss_type == "column_entropy":
        columns = 1 / T * scores.sum(dim=0)
        entropy_columns = _entropy(columns)
        return -entropy_columns


if __name__ == "__main__":
    gate = TopKGate(network=nn.Softmax(), balancing_loss_type="entropy")
    x = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=float, requires_grad=True)
    y = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]])
    criterion = nn.CrossEntropyLoss()
    topk = gate(x)
    loss = criterion(topk[1], y.float())
    loss.backward()
    print(x.grad)
