import os
import wandb
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from typing import List

wandb.init(project="robust-cifar100-resnet-moe")

api = wandb.Api()

all_runs: List[wandb.wandb_sdk.wandb_run.Run] = api.runs("robust-cifar100-resnet-moe")

class SimpleTable:
    def __init__(self, columns, data):
        self.columns = columns
        self.data = data
        
    def get_column(self, col_name):
        idx = self.columns.index(col_name)
        return [row[idx] for row in self.data]

def get_table(artifact_run, table_names):
    if isinstance(table_names, str):
        table_names = [table_names]
    run_id = artifact_run.id

    for table_name in table_names:
        short_name = table_name.replace("-", "")
        artifact_ref = f"run-{run_id}-{table_name}:latest"
        try:
            artifact = wandb.use_artifact(artifact_ref)
            local_dir = artifact.download()
            json_path = os.path.join(local_dir, f"{table_name}.table.json")
            
            with open(json_path, "r") as f:
                raw = json.load(f)
            
            return SimpleTable(columns=raw["columns"], data=raw["data"])
        except Exception as e:
            print(f"Error loading table {table_name}: {e}")

    raise ValueError("None of the given tables could be found!")

def load_fixed_expert_table(table, column="Metric"):
    accs = table.get_column(column)
    all_experts_acc = accs[-1]  # Last row is overall performance
    fe_accs = accs[:-1]  # Previous rows are individual fixed expert performance
    return all_experts_acc, fe_accs

def load_expert_accs(runs: list, table_names="performance_plot_natural_table", column="Metric"):
    accs = []
    fe_accs_list = []
    for run in runs:
        try:
            table = get_table(run, table_names)
            all_experts_acc, fe_accs = load_fixed_expert_table(table, column=column)
            accs.append(all_experts_acc)
            fe_accs_list.append(fe_accs)
            print(f"Successfully loaded {run.name}: Overall accuracy={all_experts_acc:.6f}, Expert count={len(fe_accs)}")
        except Exception as e:
            print(f"Failed to load {run.name}: {e}")
    return accs, fe_accs_list

def match_fixed_expert_accuracies(num_experts_list, fe_accs_list):
    """Fixed function: correctly match expert count and fixed expert data"""
    fe_data = []
    for num_experts, fe_accs in zip(num_experts_list, fe_accs_list):
        # Each expert count corresponds to its fixed expert count
        for expert_idx, fe_acc in enumerate(fe_accs):
            fe_data.append((num_experts, fe_acc))
    return np.array(fe_data, dtype=float)

def plot_fixed_experts(
    num_experts_list,
    accs,
    fe_data_np,
    fe_data_brown_np=None,
    show_accs=True,
    show_fe_scatter=True,
    show_sota=False,
    ylabel="Adversarial Accuracy",
    baseline=0.178,
):
    np.random.seed(1)

    plt.figure(figsize=(10, 6))
    plt.xticks(num_experts_list)
    plt.ylabel(ylabel)
    plt.xlabel("Number of Experts")

    legend = ["ResNet18 Baseline"]
    plt.axhline(baseline, color="green", linestyle="--")
    (accs_line,) = plt.plot(num_experts_list, accs, marker="x", linewidth=2, markersize=8)

    if show_accs:
        legend.append("ResNet18-BlockMoE; k=1")
    else:
        accs_line.remove()

    if show_fe_scatter and len(fe_data_np) > 0:
        fe_data_np = np.copy(fe_data_np)
        fe_data_np[:, 0] += fe_data_np[:, 0] * np.random.uniform(
            -0.1, 0.1, size=fe_data_np.shape[0]
        )
        plt.scatter(fe_data_np[:, 0], fe_data_np[:, 1], color="brown", alpha=0.6, s=50)
        legend.append("Fixed Expert (robust)")

    if show_fe_scatter and fe_data_brown_np is not None and len(fe_data_brown_np) > 0:
        fe_data_brown_np = np.copy(fe_data_brown_np)
        fe_data_brown_np[:, 0] += fe_data_brown_np[:, 0] * np.random.uniform(
            -0.1, 0.1, size=fe_data_brown_np.shape[0]
        )
        plt.scatter(fe_data_brown_np[:, 0], fe_data_brown_np[:, 1], alpha=0.6, s=50)
        legend.append("Fixed Expert")

    if show_sota:
        plt.axhspan(0.25, 0.27, color="red", linestyle="--", alpha=0.3)
        legend.append("ResNet18 SOTA")

    plt.legend(legend)
    plt.semilogx(base=2)
    plt.grid(True, alpha=0.3)

def fixed_expert_performance_plots(runs, figure_prefix, baseline_natural, baseline_attacked):
    print(f"\nProcessing {len(runs)} runs...")
    
    print("Loading natural accuracy data...")
    accs_list, fe_accs_list = load_expert_accs(
        runs, table_names=("performance_plot_natural_table")
    )
    
    print("Loading adversarial accuracy data...")
    accs_robust_list, fe_accs_robust_list = load_expert_accs(
        runs, table_names=("performance_plot_PGD-20-8-2_table")
    )
    
    num_experts_list = [2, 4, 8, 16, 32]
    
    fe_data_np = match_fixed_expert_accuracies(num_experts_list, fe_accs_list)
    fe_data_robust_np = match_fixed_expert_accuracies(num_experts_list, fe_accs_robust_list)
    
    print(f"Natural accuracy data shape: {fe_data_np.shape}")
    print(f"Adversarial accuracy data shape: {fe_data_robust_np.shape}")
    
    print("\n=== Validating data correspondence ===")
    for i, (num_experts, fe_accs) in enumerate(zip(num_experts_list, fe_accs_list)):
        print(f"Expert count {num_experts}: {len(fe_accs)} fixed experts")
    
    print("\n=== Computing color classification ===")
    
    fe_data_robust_up = []
    fe_data_robust_down = []
    
    fe_data_up = []
    fe_data_down = []
    
    current_idx = 0
    for i, (num_experts, fe_accs_robust, fe_accs_natural) in enumerate(zip(num_experts_list, fe_accs_robust_list, fe_accs_list)):
        moe_overall_robust = accs_robust_list[i]  # MoE model overall adversarial accuracy
        moe_overall_natural = accs_list[i]  # MoE model overall natural accuracy
        
        print(f"Expert count {num_experts}:")
        print(f"  MoE overall adversarial accuracy = {moe_overall_robust:.6f}")
        print(f"  MoE overall natural accuracy = {moe_overall_natural:.6f}")
        
        for j, (fe_acc_robust, fe_acc_natural) in enumerate(zip(fe_accs_robust, fe_accs_natural)):
            # Adversarial accuracy classification
            if fe_acc_robust > moe_overall_robust:
                fe_data_robust_up.append([num_experts, fe_acc_robust])
                print(f"  Expert {j}: Adversarial {fe_acc_robust:.6f} > {moe_overall_robust:.6f} -> brown")
            else:
                fe_data_robust_down.append([num_experts, fe_acc_robust])
                print(f"  Expert {j}: Adversarial {fe_acc_robust:.6f} <= {moe_overall_robust:.6f} -> normal")
            
            # Natural accuracy classification
            if fe_acc_natural > moe_overall_natural:
                fe_data_up.append([num_experts, fe_acc_natural])
                print(f"  Expert {j}: Natural {fe_acc_natural:.6f} > {moe_overall_natural:.6f} -> brown")
            else:
                fe_data_down.append([num_experts, fe_acc_natural])
                print(f"  Expert {j}: Natural {fe_acc_natural:.6f} <= {moe_overall_natural:.6f} -> normal")
    
    fe_data_robust_up = np.array(fe_data_robust_up) if fe_data_robust_up else np.empty((0, 2))
    fe_data_robust_down = np.array(fe_data_robust_down) if fe_data_robust_down else np.empty((0, 2))
    fe_data_up = np.array(fe_data_up) if fe_data_up else np.empty((0, 2))
    fe_data_down = np.array(fe_data_down) if fe_data_down else np.empty((0, 2))
    
    print(f"\nAdversarial accuracy plot:")
    print(f"  Brown point count: {len(fe_data_robust_up)}")
    print(f"  Normal point count: {len(fe_data_robust_down)}")
    print(f"\nNatural accuracy plot:")
    print(f"  Brown point count: {len(fe_data_up)}")
    print(f"  Normal point count: {len(fe_data_down)}")

    os.makedirs(os.path.dirname(f"{figure_prefix}_attacked_fixed_expert_plot.pdf"), exist_ok=True)

    print("Plotting adversarial accuracy...")
    plot_fixed_experts(
        num_experts_list,
        accs_robust_list,
        fe_data_robust_up,
        fe_data_brown_np=fe_data_robust_down,
        show_fe_scatter=True,
        show_sota=False,
        baseline=baseline_attacked,
    )
    plt.savefig(f"{figure_prefix}_attacked_fixed_expert_plot.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print("Plotting natural accuracy...")
    plot_fixed_experts(
        num_experts_list,
        accs_list,
        fe_data_up,
        fe_data_brown_np=fe_data_down,
        show_fe_scatter=True,
        show_sota=False,
        ylabel="Accuracy",
        baseline=baseline_natural,
    )
    plt.savefig(f"{figure_prefix}_natural_fixed_expert_plot.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    num_experts_list = [2, 4, 8, 16, 32]

    training_type = "adv-train-CGARN"  # Switch to "adv-train" to use adversarial training results

    if training_type == "adv-train-CGARN":
        run_names = [f"evaluate-cifar100-resnet18-pgd-adv-train-block-moe{ne}-CGARN-1-switch" for ne in num_experts_list]
    else:
        run_names = [f"evaluate-cifar100-resnet18-block-moe{ne}-CGARN-1-switch" for ne in num_experts_list]


    natural_runs = []
    for run_name in run_names:
        run = None
        for r in all_runs:
            if r.name == run_name:
                run = r
                break
        if run:
            natural_runs.append(run)

    def get_expert_count(run):
        return int(run.name.split("moe")[1].split("-")[0])

    natural_runs.sort(key=get_expert_count)

    print(f"Found {len(natural_runs)} matching runs:")
    for i, run in enumerate(natural_runs):
        expert_count = get_expert_count(run)
        print(f"  {i}: {run.name} (ID: {run.id}, Expert count: {expert_count})")

    adv_trained_baseline_natural = 0.5214
    adv_trained_baseline_attacked = 0.1687
    normal_trained_baseline_natural = 0.7130
    normal_trained_baseline_attacked = 0.0006

    if len(natural_runs) > 0:
        figure_prefix = f"fixed_expert_plots/switch_CGARN/cifar-switch-{training_type}"

        if training_type == "adv-train-CGARN":
            baseline_natural = adv_trained_baseline_natural
            baseline_attacked = adv_trained_baseline_attacked
        else:
            baseline_natural = normal_trained_baseline_natural
            baseline_attacked = normal_trained_baseline_attacked

        fixed_expert_performance_plots(
            natural_runs,
            figure_prefix,
            baseline_natural=baseline_natural,
            baseline_attacked=baseline_attacked
        )
    else:
        print("No matching runs found!") 