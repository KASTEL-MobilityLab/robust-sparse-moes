import copy
from typing import Any, Tuple, Union, cast

import torch
import torch.nn.functional as F
from einops import einops
from torch import Tensor, nn
from torch.nn import Module, ModuleList

from src.models.nn.moe.gate.topk import TopKGate


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
    from src.models.nn.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d

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


class MOELayer(torch.nn.Module):
    """MOELayer module which implements a simple expert routing for one device."""

    def __init__(
        self, gate: Module, experts: Union[Module, ModuleList], var_alpha: float = 0
    ) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])

        self.num_local_experts = len(self.experts)
        self.fixed_expert = None
        self.variance_alpha = var_alpha

    def _compute_variance_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Compute loss based on variance/covariance of the output.

        Args:
            expert_outputs: ExCxHxW tensor
        """

        # Encourage variance!
        std = torch.sqrt(expert_outputs.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std))

        return std_loss

    def forward(self, input: Tensor, **kwargs: Any) -> Tensor:
        if self.fixed_expert not in {None, -1, self.num_local_experts}:
            return self.experts[self.fixed_expert](input)

        # Expert routing:
        # shape=(T,E)
        _, expert_routing = self.gate(input)

        outputs = []
        for i, expert in enumerate(self.experts):
            selected_tokens = expert_routing[:, i]
            selected_inputs = input[selected_tokens != 0]
            if selected_inputs.shape[0] > 0:
                outputs.append(expert(selected_inputs))
            else:
                outputs.append(None)

        batch_size = input.shape[0]
        num_experts = len(self.experts)
        expert_output_shape = None
        dtype = None
        for output in outputs:
            if output is not None:
                expert_output_shape = output.shape[1:]
                dtype = output.dtype
                break

        out = torch.zeros(
            (num_experts, batch_size, *expert_output_shape),
            dtype=dtype,
            device=input.device,
        )

        for i, output in enumerate(outputs):
            if output is not None:
                selected_tokens = expert_routing[:, i]
                out[i, selected_tokens != 0] += output

        out = einops.rearrange(out, "e b ... -> e b (...)")
        out *= expert_routing.T[:, :, None]
        out = einops.reduce(out, "e b ... -> b ...", "sum")
        out = torch.reshape(out, (batch_size, *expert_output_shape))

        if self.variance_alpha > 0:
            mean_outputs = [
                (output.mean() if output else torch.tensor(0, dtype=out.dtype, device=out.device))
                for output in outputs
            ]
            expert_mean_outputs = torch.stack(mean_outputs)
            self.loss = self.variance_alpha * self._compute_variance_loss(expert_mean_outputs)

        return out


class SkipMOELayer(MOELayer):
    """MOELayer module which adds expert output to the input.

    Adding this residual connection ensures gradient flow towards the gating network.
    An experts influence can be reduced/increased.

    This residual skip connection might already be part of the models architecture however!
    """

    def forward(self, input: Tensor, **kwargs: Any) -> Tensor:
        out = super().forward(input, **kwargs)

        assert (
            input.shape == out.shape
        ), f"{self.__class__} requires experts to keep the shape of the inputs."
        return input + out


def patchify(x: torch.Tensor, ps_channel, ps_vert, ps_hor):
    return einops.rearrange(
        x,
        "b (patches_c ps_channel) (patches_vert ps_vert) (patches_hor ps_hor) -> (b patches_c patches_vert "
        "patches_hor) ps_channel ps_vert ps_hor",
        ps_channel=ps_channel,
        ps_vert=ps_vert,
        ps_hor=ps_hor,
    )


@torch.jit.script
def patchify_script(x: torch.Tensor, ps_vert: int, ps_hor: int):
    num_patch_vert = x.shape[2] // ps_vert
    num_patch_hor = x.shape[3] // ps_hor

    x = F.pad(x, (1, 1, 1, 1), "constant", 0.0)

    return (
        x.unfold(2, ps_vert + 2, ps_vert)
        .unfold(3, ps_hor + 2, ps_hor)
        .reshape(x.shape[0] * num_patch_vert * num_patch_hor, x.shape[1], ps_vert + 2, ps_hor + 2)
    )


def unpatchify(x, h, w, ps_vert, ps_hor):
    return einops.rearrange(
        x,
        "(b patches_vert patches_hor) c ps_vert ps_hor -> b c (patches_vert ps_vert) (patches_hor ps_hor)",
        patches_vert=h // ps_vert,
        patches_hor=w // ps_hor,
    )


@torch.jit.script
def unpatchify_script(
    x: torch.Tensor, h: int, w: int, ps_vert: int, ps_hor: int, padding: bool = True
):
    # Drop padding around patches!
    if padding:
        x = x[:, :, 1:-1, 1:-1]

    num_patch_vert = h // ps_vert
    num_patch_hor = w // ps_hor

    c = x.shape[1]

    patches = x.reshape(-1, c, num_patch_vert * num_patch_hor, ps_vert * ps_hor)
    patches = patches.permute(0, 1, 3, 2)
    patches = patches.contiguous().view(-1, c * ps_vert * ps_hor, num_patch_hor * num_patch_vert)

    return F.fold(
        patches, output_size=[h, w], kernel_size=[ps_vert, ps_hor], stride=[ps_vert, ps_hor]
    )


class PatchMOELayer(MOELayer):
    """Inputs are distributed into small patches in either only the (h,w) dimensions or in the
    (c,h,w) dimensions."""

    def __init__(self, *args, patch_size: Tuple[int, int], **kwargs):
        if len(patch_size) == 2:
            patch_size = (0, *patch_size)
        self.patch_size = patch_size

        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        ps_channel, ps_vert, ps_hor = self.patch_size

        # Padding
        _, original_c, original_h, original_w = x.shape

        if ps_channel == 0:
            ps_channel = original_c

        # No padding for channel because it's input independent and can be chosen easily a priori
        v_pad = ps_vert - original_h % ps_vert
        h_pad = ps_hor - original_w % ps_hor
        x = F.pad(x, (0, h_pad, 0, v_pad), value=0)

        # Ensure shapes match up
        b, c, h, w = x.shape
        assert c % ps_channel == 0
        assert h % ps_vert == 0
        assert w % ps_hor == 0

        # x = patchify(x, ps_channel, ps_vert, ps_hor)
        patched_x = patchify_script(x, ps_vert, ps_hor)
        patched_x = super().forward(patched_x, **kwargs)
        x = unpatchify_script(patched_x, h, w, ps_vert, ps_hor)
        # x = unpatchify(x, h, w, ps_vert, ps_hor)

        if ps_channel != original_c:
            # The channel wise patching expects that the inputs are patched into multiple channel segments,
            # the outputs of the layer span over the whole channel space
            # --> The outputs must be summed up over all patches
            x = einops.rearrange(
                x, "(b patches_c) c h w -> b patches_c c h w", patches_c=c // ps_channel
            )
            x = einops.reduce(x, "b patches_c c h w -> b c h w", reduction="sum")

        # Reverse padding
        x = x[:, :, :original_h, :original_w]

        return x


class ResidualMOELayer(MOELayer):
    """MOELayer module which adds expert output to the input.

    Adding this residual connection ensures gradient flow towards the gating network.
    An experts influence can be reduced/increased.

    This residual skip connection might already be part of the models architecture however!
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList]) -> None:
        self.residual_expert = copy.deepcopy(experts[0])
        _initialize_weights(self.residual_expert)
        super().__init__(gate, experts)

    def forward(self, input: Tensor, **kwargs: Any) -> Tensor:
        out = self.forward(input, **kwargs)

        return self.residual_expert(input) + out


class SkipPatchMoE(SkipMOELayer, PatchMOELayer):
    pass


class ResidualPatchMOE(ResidualMOELayer, PatchMOELayer):
    pass


if __name__ == "__main__":
    import torch

    # x = torch.randn((1, 1, 4, 8))
    # print((x, x.shape))
    # y = patchify_script(x, 2, 2)
    # print((y, y.shape))
    # z = unpatchify_script(y, 4, 8, 2, 2)
    # print((z, z.shape))
    # assert (x == z).all()
    gate = TopKGate(
        network=nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=(2,)), nn.Linear(2, 2), nn.Softmax()
        )
    )
    layer = MOELayer(experts=nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)]), gate=gate)
    layer.train()

    x = torch.tensor([[1, 2], [1, 2]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[0, 0], [0, 1]])
    criterion = nn.MSELoss()
    out = layer(x)
    out.abs().sum().backward()
    assert layer.experts[0].weight.grad is not None or layer.experts[1].weight.grad is not None
