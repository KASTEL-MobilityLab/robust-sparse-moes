import logging

import torch.nn

logger = logging.getLogger(__name__)


class AbstractGate(torch.nn.Module):
    pass


def topk_mask(input, k, dim=0, replacement: float = 0):
    assert len(input.shape) == 2

    n = input.shape[dim % 2]
    q = 1 - k / n

    quant = torch.quantile(input, q=q, dim=dim, keepdim=True)

    out = (input > quant) * input
    out[input <= quant] = replacement

    return out


class RandomTopKGate(AbstractGate):
    def __init__(self, num_experts: int, in_features: int, k=2):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.k = k

    def forward(self, input: torch.Tensor):
        num_input_tokens = input.shape[0]
        routing = torch.randn((num_input_tokens, self.num_experts), device=input.device)
        mask = topk_mask(routing, self.k, dim=1)
        return (mask / (mask + 1e-2)).round()


class SimpleTopKGate(AbstractGate):
    def __init__(
        self, num_experts: int, in_features: int, k=2, expert_capacity: float = float("inf")
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.q = 1 - self.k / self.num_experts

        self.max_relative_tokens = expert_capacity * self.k / self.num_experts
        if self.max_relative_tokens > 1:
            self.max_relative_tokens = float("inf")

        self.linear = torch.nn.Linear(in_features=in_features, out_features=num_experts)

    def _capacity_mask(self, input) -> torch.Tensor:
        EPS = 1e-8
        if self.max_relative_tokens == float("inf"):
            return input

        batch_size = input.shape[0]
        max_tokens = max(int(batch_size * self.max_relative_tokens), 1)

        output = torch.softmax(input, dim=1)
        output = topk_mask(output, max_tokens, dim=0, replacement=EPS)

        output = torch.log(output)
        # print(output[0])

        # assert (output[:, 0] > math.log(EPS)).sum() == max_tokens

        return output

    def forward(self, input: torch.Tensor):
        HIGH_NEGATIVE = -1e5
        flat_input = torch.flatten(input, start_dim=1)

        output = self.linear(flat_input)

        # TODO: Add noise
        output = output + torch.normal(0, std=0.1, size=output.shape, device=output.device)

        # Enforce capacity
        if self.max_relative_tokens != float("inf"):
            batch_size = output.shape[0]
            max_tokens = max(int(batch_size * self.max_relative_tokens), 1)
            capacity_mask = topk_mask(output, k=max_tokens, dim=0, replacement=HIGH_NEGATIVE)
            output_no_capacity = output.clone()
            output[capacity_mask == HIGH_NEGATIVE] = HIGH_NEGATIVE

        # enforce topk
        output = topk_mask(output, self.k, dim=1, replacement=HIGH_NEGATIVE)

        # Apply soft max
        output = torch.softmax(output, dim=1)

        activated_exp = (output[0] != 0).sum()
        if activated_exp < self.k:
            logger.warning("Hard gate returning <k non-zero entries!")
        elif activated_exp > self.k:
            logger.error(
                f"Hard gate returning {(output[0] != 0).sum()}>k={self.k} non-zero entries!"
            )

        return output
