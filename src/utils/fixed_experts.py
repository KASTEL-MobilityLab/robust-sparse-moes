import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb


def load_fixed_expert_table(table, column="Metric"):
    accs = table.get_column(column)
    all_experts_acc = accs[-1]
    fe_accs = accs[:-1]
    return all_experts_acc, fe_accs


def get_table(artifact_run, table_names):
    if isinstance(table_names, str):
        table_names = [table_names]
    id = artifact_run.id
    for table_name in table_names:
        try:
            short_table_name = table_name.replace("-", "")
            my_table = wandb.use_artifact(f"run-{id}-{short_table_name}:v0").get(
                f"{table_name}.table.json"
            )
            return my_table
        except Exception as e:
            print(f"Ignoring error: {e}")
    raise ValueError("None of the given tables could be found!")


def load_expert_accs(runs: list, table_names="loss_plot_PGD-20-8-2_table", column="Metric"):
    accs = []
    fe_accs_list = []
    for run in runs:
        table = get_table(run, table_names)
        all_experts_acc, fe_accs = load_fixed_expert_table(table, column=column)
        accs.append(all_experts_acc)
        fe_accs_list.append(fe_accs)
    return accs, fe_accs_list


def match_fixed_expert_accuracies(num_experts_list, fe_accs_list):
    fe_data = [
        (num_experts, fe_acc)
        for num_experts, fe_accs in zip(num_experts_list, fe_accs_list)
        for fe_acc in fe_accs
    ]
    return np.array(fe_data, dtype=float)


def plot_fixed_experts(
    num_experts_list,
    accs,
    fe_data_np,
    fe_data_brown_np=None,
    show_accs=True,
    show_fe_scatter=True,
    show_sota=False,
    ylabel="Adversarial mIoU",
    xlabel="Number of Experts",
    baseline=0.1777,
):
    np.random.seed(1)

    plt.xticks(num_experts_list)
    # plt.xlim(1.5,max(num_experts_list)+1)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    legend = []
    if baseline is not None:
        legend.append("Baseline")
        plt.axhline(baseline, color="green", linestyle="--")

    (accs_line,) = plt.plot(num_experts_list, accs, marker="x")

    if show_accs:
        legend.append("PatchConvMoE; k=1")
    else:
        accs_line.remove()

    if show_fe_scatter and len(fe_data_np) > 0:
        fe_data_np = np.copy(fe_data_np)
        fe_data_np[:, 0] += fe_data_np[:, 0] * np.random.uniform(
            -0.1, 0.1, size=fe_data_np.shape[0]
        )
        sns.scatterplot(x=fe_data_np[:, 0], y=fe_data_np[:, 1], color="brown")
        legend.append("Fixed Expert (robust)")

    if show_fe_scatter and fe_data_brown_np is not None and len(fe_data_brown_np) > 0:
        fe_data_brown_np = np.copy(fe_data_brown_np)
        fe_data_brown_np[:, 0] += fe_data_brown_np[:, 0] * np.random.uniform(
            -0.1, 0.1, size=fe_data_brown_np.shape[0]
        )
        sns.scatterplot(x=fe_data_brown_np[:, 0], y=fe_data_brown_np[:, 1])
        legend.append("Fixed Expert")

    plt.legend(legend)
    plt.semilogx(base=2)
    # plt.ylim(0.05, 0.3)


def fixed_expert_performance_plots(
    runs,
    figure_prefix,
    baseline_natural,
    baseline_attacked,
    num_experts_list,
    metric_name="Accuracy",
    highlight_robust=True,
    xlabel="Number of Experts",
):
    accs_list, fe_accs_list = load_expert_accs(
        runs, table_names=("performance_plot_natural_table", "loss_plot_natural_table")
    )
    accs_robust_list, fe_accs_robust_list = load_expert_accs(
        runs, table_names=("performance_plot_PGD-20-8-2_table", "loss_plot_PGD-20-8-2_table")
    )
    fe_data_np = match_fixed_expert_accuracies(num_experts_list, fe_accs_list)
    fe_data_robust_np = match_fixed_expert_accuracies(num_experts_list, fe_accs_robust_list)

    accs_robust_np = np.array(
        [accs_robust_list[num_experts_list.index(x)] for x in fe_data_robust_np[:, 0]]
    )
    accs_robust_up = fe_data_robust_np[:, 1] >= accs_robust_np
    if not highlight_robust:
        accs_robust_up[:] = 0
    fe_data_robust_up = fe_data_robust_np[accs_robust_up]
    fe_data_robust_down = fe_data_robust_np[~accs_robust_up]

    fe_data_up = fe_data_np[accs_robust_up]
    fe_data_down = fe_data_np[~accs_robust_up]

    plot_fixed_experts(
        num_experts_list,
        accs_robust_list,
        fe_data_robust_up,
        fe_data_brown_np=fe_data_robust_down,
        show_fe_scatter=True,
        show_sota=False,
        baseline=baseline_attacked,
        ylabel="Adversarial " + metric_name,
        xlabel=xlabel,
    )
    plt.savefig(f"{figure_prefix}_adv_fixed_expert_plot.png")
    plt.show()

    plot_fixed_experts(
        num_experts_list,
        accs_list,
        fe_data_up,
        fe_data_brown_np=fe_data_down,
        show_fe_scatter=True,
        show_sota=False,
        ylabel=metric_name,
        baseline=baseline_natural,
        xlabel=xlabel,
    )
    plt.savefig(f"{figure_prefix}_natural_fixed_expert_plot.png")
    plt.show()
