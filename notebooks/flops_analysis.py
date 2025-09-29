import os
import sys
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.models.nn.moe.resnet_block_moe import ResNetBlockMoE
from src.models.nn.moe.routing import GlobalAvgLinearRoutingNetwork
from src.models.nn.moe.resnet_conv_moe import ResNetConvMoE
import torchinfo
from torchinfo import summary

os.chdir('/...')
print(os.getcwd())

MOE_TYPE = ResNetBlockMoE

def get_models(k=1):
    models = [
        MOE_TYPE(
            routing_layer_type=GlobalAvgLinearRoutingNetwork,
            layers=18,
            num_experts=ne,
            k=k if k >= 1 else max(round(ne * k), 1),
        )
        for ne in (1, 2, 4, 8, 16, 32)
    ]
    return models

def multi_adds(k=1):
    models = get_models(k)
    return [summary(model, input_size=(1, 1, 28, 28), device='cpu').total_mult_adds for model in models]

num_experts_list = [1, 2, 4, 8, 16, 32]

flops_k_experts = multi_adds(k=0.99)  # k = #Experts
flops_k_half = multi_adds(k=0.5)      # k = 0.5 #Experts  
flops_k_2 = multi_adds(k=2)           # k = 2
flops_k_1 = multi_adds(k=1)           # k = 1

plt.figure(figsize=(10, 6))

plt.plot(num_experts_list, flops_k_experts, 'o-', color='blue', linewidth=3, markersize=8, label='k = #Experts')
plt.plot(num_experts_list, flops_k_half, 'o-', color='orange', linewidth=3, markersize=8, label='k = 0.5 #Experts')
plt.plot(num_experts_list, flops_k_2, 'o-', color='green', linewidth=3, markersize=8, label='k = 2')
plt.plot(num_experts_list, flops_k_1, 'o-', color='red', linewidth=3, markersize=8, label='k = 1')
plt.plot(num_experts_list, flops_k_2, 'o-', color='green', linewidth=3, markersize=8, label='k = 2')
plt.plot(num_experts_list, flops_k_1, 'o-', color='red', linewidth=3, markersize=8, label='k = 1')

plt.xlabel('Number of Experts', fontsize=18)
plt.ylabel('FLOPS', fontsize=18)
plt.title('Number of Experts vs FLOPS', fontsize=18, fontweight='bold')

plt.xscale('log')
plt.yscale('log')

plt.xticks(num_experts_list, num_experts_list, fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, alpha=0.3)

plt.legend(fontsize=18, loc='upper left')

plt.tight_layout()

plt.savefig('flops_vs_experts.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as flops_vs_experts.pdf")
