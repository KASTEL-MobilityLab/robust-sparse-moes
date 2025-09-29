#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness-Accuracy Trade-off Visualization Script
For ResNet18 and ResNet50
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_moe_results(model_type='ResNet18', moe_type='ConvMoE', attack_type='PGD'):
    """
    Args:
        model_type: 'ResNet18' or 'ResNet50', select the model to plot
        moe_type: 'ConvMoE' or 'BlockMoE', select the MoE type to plot
        attack_type: 'PGD' or 'AutoPGD', select the evaluation method
    """
    print(f"Generating {attack_type} evaluation results for {moe_type} of {model_type}...")
    print("Preparing data...")
    
    moe_data_resnet18 = {
        ("BlockMoE", "L_ent", "Conv-GAP"): {
            1: {"normal": (70.08, 0.01, 2.22), "adversarial": (51.97, 19.06, 22.11)},
            2: {"normal": (66.54, 0.02, 2.89), "adversarial": (50.69, 17.79, 19.45)},
            3: {"normal": (66.16, 0.01, 2.48), "adversarial": (50.93, 17.41, 18.99)},
            4: {"normal": (64.96, 0.10, 2.65), "adversarial": (50.63, 17.23, 18.81)},
        },
        ("BlockMoE", "L_ent", "GAP-FC"): {
            1: {"normal": (70.20, 0.00, 2.66), "adversarial": (51.76, 19.75, 22.31)},
            2: {"normal": (67.93, 0.01, 2.57), "adversarial": (53.34, 18.40, 20.38)},
            3: {"normal": (67.90, 0.05, 2.73), "adversarial": (51.89, 18.10, 19.96)},
            4: {"normal": (68.05, 0.00, 2.42), "adversarial": (51.77, 18.47, 19.20)},
        },
        ("BlockMoE", "L_switch", "Conv-GAP"): {
            1: {"normal": (70.19, 0.03, 2.51), "adversarial": (50.82, 18.33, 20.99)},
            2: {"normal": (71.94, 0.00, 2.96), "adversarial": (50.94, 18.75, 20.25)},
            3: {"normal": (71.12, 0.07, 2.23), "adversarial": (50.61, 18.42, 19.82)},
            4: {"normal": (72.16, 0.00, 2.73), "adversarial": (51.00, 17.57, 18.31)},
        },
        ("BlockMoE", "L_switch", "GAP-FC"): {
            1: {"normal": (70.86, 0.03, 2.99), "adversarial": (51.36, 22.69, 21.22)},
            2: {"normal": (72.23, 0.01, 2.34), "adversarial": (51.39, 18.03, 20.01)},
            3: {"normal": (71.64, 0.01, 2.34), "adversarial": (52.06, 18.19, 19.92)},
            4: {"normal": (72.31, 0.00, 2.56), "adversarial": (51.72, 17.85, 18.75)},
        },
        ("BlockMoE", "L_column_entropy", "Conv-GAP"): {
            1: {"normal": (70.04, 0.00, 3.18), "adversarial": (49.55, 17.39, 20.95)},
            2: {"normal": (72.57, 0.00, 2.93), "adversarial": (50.60, 17.70, 18.59)},
            3: {"normal": (71.97, 0.01, 2.73), "adversarial": (50.21, 19.95, 19.17)},
            4: {"normal": (72.74, 0.01, 2.49), "adversarial": (50.60, 19.70, 19.59)},
        },
        ("BlockMoE", "L_column_entropy", "GAP-FC"): {
            1: {"normal": (69.91, 0.01, 3.43), "adversarial": (51.15, 18.84, 21.22)},
            2: {"normal": (69.32, 0.02, 2.35), "adversarial": (48.79, 19.38, 20.13)},
            3: {"normal": (70.35, 0.01, 2.59), "adversarial": (50.25, 19.29, 19.27)},
            4: {"normal": (70.80, 0.07, 2.79), "adversarial": (51.38, 19.53, 17.77)},
        },
        ("ConvMoE", "L_ent", "Conv-GAP"): {
            1: {"normal": (68.11, 0.00, 2.89), "adversarial": (50.87, 22.74, 22.46)},
            2: {"normal": (44.33, 0.76, 4.23), "adversarial": (24.96, 14.46, 16.13)},
            3: {"normal": (45.40, 1.05, 3.55), "adversarial": (22.14, 13.20, 15.32)},
            4: {"normal": (38.61, 1.30, 3.01), "adversarial": (25.99, 14.77, 16.85)},
        },
        ("ConvMoE", "L_ent", "GAP-FC"): {
            1: {"normal": (69.84, 0.00, 3.11), "adversarial": (50.96, 25.03, 24.03)},
            2: {"normal": (45.64, 2.55, 3.23), "adversarial": (29.36, 16.51, 18.28)},
            3: {"normal": (43.85, 0.04, 3.89), "adversarial": (27.04, 15.17, 16.89)},
            4: {"normal": (28.26, 1.34, 4.03), "adversarial": (30.88, 17.33, 19.27)},
        },
        ("ConvMoE", "L_switch", "Conv-GAP"): {
            1: {"normal": (68.93, 0.04, 3.26), "adversarial": (49.15, 19.41, 20.51)},
            2: {"normal": (71.54, 0.01, 3.45), "adversarial": (47.89, 18.61, 19.26)},
            3: {"normal": (70.72, 0.06, 2.89), "adversarial": (50.21, 19.95, 19.17)},
            4: {"normal": (71.45, 0.00, 3.88), "adversarial": (50.60, 19.70, 18.59)},
        },
        ("ConvMoE", "L_switch", "GAP-FC"): {
            1: {"normal": (69.89, 0.00, 3.28), "adversarial": (49.08, 18.85, 20.40)},
            2: {"normal": (71.32, 0.02, 3.35), "adversarial": (48.79, 19.38, 20.13)},
            3: {"normal": (71.35, 0.01, 2.59), "adversarial": (50.25, 19.29, 19.27)},
            4: {"normal": (71.80, 0.07, 3.79), "adversarial": (51.38, 19.53, 17.77)},
        },
        ("ConvMoE", "L_column_entropy", "Conv-GAP"): {
            1: {"normal": None, "adversarial": (49.55, 17.39, 20.95)},
            2: {"normal": None, "adversarial": None},
            3: {"normal": None, "adversarial": None},
            4: {"normal": None, "adversarial": None},
        },
        ("ConvMoE", "L_column_entropy", "GAP-FC"): {
            1: {"normal": None, "adversarial": (49.63, 17.32, 21.95)},
            2: {"normal": None, "adversarial": None},
            3: {"normal": None, "adversarial": None},
            4: {"normal": None, "adversarial": None},
        },
        ("Baseline", None, "Baseline"): {
            "Baseline": {"normal": (71.30, 0.06, 2.31), "adversarial": (52.14, 16.87, 17.99)},
        },
    }

    moe_data_resnet50 = {
        ("Baseline", None, "Baseline"): {
            "Baseline": {"normal": (73.18, 0.06, 1.87), "adversarial": (53.25, 17.21, 1.36)},
        },
        ("BlockMoE", "L_ent", "Conv-GAP"): {
            1: {"normal": (72.29, 0.02, 2.71), "adversarial": (52.70, 20.76, 1.20)},
            2: {"normal": (68.38, 0.28, 2.47), "adversarial": (43.31, 18.86, 0.97)},
            3: {"normal": (68.30, 0.34, 1.83), "adversarial": (41.82, 18.87, 1.06)},
            4: {"normal": (68.33, 0.44, 1.74), "adversarial": (40.32, 19.03, 1.15)},
        },
        ("BlockMoE", "L_ent", "GAP-FC"): {
            1: {"normal": (72.09, 0.04, 3.47), "adversarial": (54.05, 20.81, 1.05)},
            2: {"normal": (70.30, 0.01, 1.31), "adversarial": (46.94, 19.93, 1.01)},
            3: {"normal": (68.76, 0.22, 1.57), "adversarial": (49.29, 18.72, 1.00)},
            4: {"normal": (69.68, 0.25, 1.37), "adversarial": (48.52, 18.12, 1.00)},
        },
        ("BlockMoE", "L_switch", "Conv-GAP"): {
            1: {"normal": (72.25, 0.43, 1.83), "adversarial": (53.75, 19.01, 1.00)},
            2: {"normal": (72.93, 0.11, 1.34), "adversarial": (51.99, 19.02, 1.23)},
            3: {"normal": (72.83, 0.11, 1.93), "adversarial": (51.85, 18.27, 1.43)},
            4: {"normal": (72.44, 0.16, 1.07), "adversarial": (51.04, 19.82, 1.43)},
        },
        ("BlockMoE", "L_switch", "GAP-FC"): {
            1: {"normal": (73.34, 0.02, 3.15), "adversarial": (50.90, 17.21, 1.36)},
            2: {"normal": (72.83, 0.05, 1.63), "adversarial": (51.46, 19.95, 1.34)},
            3: {"normal": (73.04, 0.11, 1.45), "adversarial": (52.99, 20.26, 1.35)},
            4: {"normal": (72.53, 0.13, 1.26), "adversarial": (52.85, 20.26, 1.37)},
        },
        ("ConvMoE", "L_ent", "Conv-GAP"): {
            1: {"normal": (69.51, 2.10, 2.04), "adversarial": (50.04, 22.58, 1.04)},
            2: {"normal": (32.66, 0.74, 1.01), "adversarial": (6.04, 4.34, 0.93)},
            3: {"normal": (56.30, 0.03, 1.04), "adversarial": (7.57, 4.11, 0.94)},
            4: {"normal": (34.32, 0.56, 1.04), "adversarial": (12.76, 2.98, 1.23)},
        },
        ("ConvMoE", "L_ent", "GAP-FC"): {
            1: {"normal": (72.06, 0.28, 1.21), "adversarial": (46.03, 23.57, 1.00)},
            2: {"normal": (29.15, 2.92, 1.02), "adversarial": (4.93, 3.41, 0.98)},
            3: {"normal": (30.79, 2.09, 1.01), "adversarial": (6.32, 4.72, 1.00)},
            4: {"normal": (32.68, 0.29, 1.04), "adversarial": (9.28, 6.59, 0.92)},
        },
        ("ConvMoE", "L_switch", "Conv-GAP"): {
            1: {"normal": (71.01, 0.08, 1.52), "adversarial": (52.78, 18.52, 1.00)},
            2: {"normal": (71.83, 0.05, 1.71), "adversarial": None},
            3: {"normal": (72.72, 0.00, 3.20), "adversarial": None},
            4: {"normal": (72.83, 0.00, 2.50), "adversarial": None},
        },
        ("ConvMoE", "L_switch", "GAP-FC"): {
            1: {"normal": (71.57, 0.04, 1.29), "adversarial": (51.63, 17.90, 1.00)},
            2: {"normal": (72.48, 0.03, 2.76), "adversarial": (51.97, 18.64, 1.00)},
            3: {"normal": (71.70, 0.00, 2.83), "adversarial": (52.80, 18.76, 1.01)},
            4: {"normal": (72.57, 0.02, 2.24), "adversarial": (53.45, 19.52, 1.45)},
        },
    }

    moe_data = moe_data_resnet18 if model_type == 'ResNet18' else moe_data_resnet50

    data = []
    for (moe, loss, gate), kmap in moe_data.items():
        for k, tmap in kmap.items():
            for training, vals in tmap.items():
                if vals is not None:
                    data.append({
                        "moe": moe,
                        "loss": loss,
                        "gate": gate,
                        "k": k,
                        "training": training,
                        "clean": vals[0],
                        "PGD": vals[1],
                        "AutoPGD": vals[2],
                    })
    
    df = pd.DataFrame(data)
    df_selected = df[df['moe'].isin([moe_type, 'Baseline'])]

    def plot_tradeoff(df, training_type, title, metric):
        sub = df[df['training'] == training_type]
        plt.figure(figsize=(6,4))
        for _, row in sub.iterrows():
            x, y = row['clean'], row[metric]
            color = 'red' if row['moe']=='Baseline' else ('blue' if row['gate']=='GAP-FC' else 'green')
            if row['loss'] == 'L_column_entropy':
                marker = 'x'
            elif row['loss'] in (None, 'L_ent'):
                marker = 'o'
            else:
                marker = 'D'
            plt.scatter(x, y, c=color, marker=marker, s=50)
            label = 'Baseline' if row['moe']=='Baseline' else f'k={row["k"]}'
            x_range = sub['clean'].max() - sub['clean'].min()
            y_range = sub[metric].max() - sub[metric].min()
            x_offset = x_range * 0.01
            y_offset = y_range * 0.02
            plt.text(x + x_offset, y + y_offset, label, ha='center', va='bottom', fontsize=9)
        plt.title(title)
        plt.xlabel('Clean Accuracy (%)')
        plt.ylabel(f'Accuracy under {metric} Attack (%)')
        plt.grid(True)
        plt.tight_layout()

    print(f"Generating Normal Training plot...")
    plot_tradeoff(df_selected, 'normal', 'Normal Training', attack_type)
    plt.savefig(f'{model_type}_{moe_type}_Normal_Training_{attack_type}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {model_type}_{moe_type}_Normal_Training_{attack_type}.pdf")
    
    print(f"Generating Adversarial Training plot...")
    plot_tradeoff(df_selected, 'adversarial', 'Adversarial Training', attack_type)
    plt.savefig(f'{model_type}_{moe_type}_Adversarial_Training_{attack_type}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {model_type}_{moe_type}_Adversarial_Training_{attack_type}.pdf")
    
    plt.show()
    
    print("Generating legend...")
    legend_elements = [
        Line2D([0], [0], marker='o', color='red',   label='Baseline',  linestyle='', markersize=8),
        Line2D([0], [0], marker='o', color='blue',  label='GAP-FC',    linestyle='', markersize=8),
        Line2D([0], [0], marker='o', color='green', label='Conv-GAP',  linestyle='', markersize=8),
        Line2D([0], [0], marker='o', color='black',label='L_ent',     linestyle='', markersize=8),
        Line2D([0], [0], marker='D', color='black',label='L_switch',  linestyle='', markersize=8),
        Line2D([0], [0], marker='x', color='black', label='L_kl', linestyle='', markersize=8),
    ]

    fig = plt.figure(figsize=(6,1.5))
    fig.legend(handles=legend_elements, loc='center', ncol=6)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{model_type}_{moe_type}_Legend.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {model_type}_{moe_type}_Legend.pdf")
    plt.show()
    
    print("All plots have been generated and saved as PDF files!")

if __name__ == "__main__":
    plot_moe_results(model_type='ResNet18', moe_type='ConvMoE', attack_type='PGD')
    plot_moe_results(model_type='ResNet18', moe_type='ConvMoE', attack_type='AutoPGD')
    plot_moe_results(model_type='ResNet18', moe_type='BlockMoE', attack_type='PGD')
    plot_moe_results(model_type='ResNet18', moe_type='BlockMoE', attack_type='AutoPGD')

    # plot_moe_results(model_type='ResNet50', moe_type='ConvMoE', attack_type='PGD')
    # plot_moe_results(model_type='ResNet50', moe_type='ConvMoE', attack_type='AutoPGD')
    # plot_moe_results(model_type='ResNet50', moe_type='BlockMoE', attack_type='PGD')
    # plot_moe_results(model_type='ResNet50', moe_type='BlockMoE', attack_type='AutoPGD')
