"""Produce evaluation of best/worst samples w.r.t. a set of attacks.

Attacks:
* PGD
* vary number of steps
* adapt alpha (step size) inverse proportional to num steps
* e.g. 20,40,60

Metric:
* Calculate a global metric on the dataset for each attack to quantify robustness

Outputs (probably as logging output to W&B):
* {Worst, Random, Best} samples
* Corresponding outputs
*
"""
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.autograd import Variable

from src import utils
from src.models.nn.moe.gate.topk import TopKGate
from src.utils import attack
from src.utils.wandb import get_runs, load_checkpoint_file

log = utils.get_logger(__name__)


def _load_state_dict(run_name, project):
    runs = get_runs(project=project, name=run_name)

    for i, run in enumerate(runs):
        checkpoint_id = f"model-{run.id}"
        try:
            checkpoint_path = load_checkpoint_file(checkpoint_id, "model.ckpt")
            break
        except Exception:
            log.warning(
                f"There are multiple runs with this name. Could not find artifacts in the {i}th one."
            )
        log.info(f"Loaded checkpoint for run {run.id}")
    else:
        raise ValueError(f"Failed to find artifacts for run {run_name}.")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    stripped_sd = {key.replace("model.model", "model"): val for key, val in state_dict.items()}

    return stripped_sd


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_forward_hooks(child)


def setup_hooks(gate, model, lam):
    remove_all_forward_hooks(model)

    softmax = gate.network[-1]
    softmax.register_forward_hook(lam)


def compute_exp_sensitivity(
    model, gate: TopKGate, normalize, sample, num_experts=4
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    output = None

    def f(mod, inp, out):
        nonlocal output
        output = out

    setup_hooks(gate, model, lam=f)

    ce = nn.CrossEntropyLoss()
    inputs = Variable(sample.unsqueeze(0))
    inputs.requires_grad = True

    grads = []
    pred = None
    for exp in range(num_experts):
        model.eval()
        pred = model(normalize(inputs))
        goal = torch.tensor(output.shape[0] * [exp])
        print(output.shape)
        loss = ce(output, goal)
        loss.backward()
        grad = inputs.grad[0].clone().abs().sum(dim=0).cpu().numpy()
        grads.append(grad)
        inputs.grad.zero_()
        model.zero_grad()

    remove_all_forward_hooks(model)

    return grads, output[0], pred


def compute_sensitivity_to_samples(model, gate, normalize, samples, num_experts=4):
    grad_grid = []
    for sample in samples:
        grads = compute_exp_sensitivity(model, gate, normalize, sample, num_experts)
        grad_grid.append(grads)

    return grad_grid


def compute_sensitivity_to_gates(model, gates, normalize, sample, num_experts=4):
    grad_grid = []
    for gate in gates:
        grads = compute_exp_sensitivity(model, gate, normalize, sample, num_experts)
        grad_grid.append(grads)

    return grad_grid


def compute_sensitivity_to_attacks(
    model,
    normalize,
    gate,
    sample,
    target,
    steps_list=[0, 8, 16, 32, 64],
    eps=8,
    alpha=2,
    num_experts=4,
):
    new_steps = 0
    attacked_inputs = sample
    grad_grid = []
    all_inputs = []
    for idx, steps in enumerate(steps_list):
        new_steps = steps - new_steps

        if new_steps > 0:
            attacked_inputs = attack.pgd(
                model,
                normalization=normalize,
                X=attacked_inputs.view(1, *attacked_inputs.shape),
                y=target.unsqueeze(0),
                eps=eps / 255,
                alpha=alpha / 255,
                steps=new_steps,
                random_start=False,
            )[0][0]
        all_inputs.append(attacked_inputs)

        grads = compute_exp_sensitivity(model, gate, normalize, attacked_inputs, num_experts)
        grad_grid.append(grads)

    return grad_grid, all_inputs


def plot_grad_grid(inputs, grad_grid, titles) -> plt.Figure:
    fig, axs = plt.subplots(
        ncols=1 + len(grad_grid[0][0]), nrows=max(2, len(grad_grid)), sharex=True, sharey=True
    )
    fig.set_size_inches(5 * (1 + len(grad_grid[0][0])), 5 * len(grad_grid))

    for row, (input, (grads, outputs, _), title) in enumerate(zip(inputs, grad_grid, titles)):

        axs[row, 0].imshow(input.numpy().transpose(1, 2, 0))
        axs[row, 0].set_title(title)

        vmax = max(np.max(grad) for grad in grads)
        vmin = min(np.min(grad) for grad in grads)
        max_output = outputs.max()
        for col, (grad, output) in enumerate(zip(grads, outputs)):
            col += 1
            seaborn.heatmap(
                grad,
                annot=False,
                cbar=False,
                vmin=vmin,
                vmax=vmax,
                ax=axs[row, col],
                xticklabels=False,
                yticklabels=False,
            )
            axs[row, col].set_title(
                f"Weight: {output.item():.2f}" + (", MAX" if output == max_output else "")
            )

    fig.tight_layout()
    plt.show()
    return fig
