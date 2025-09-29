#!/bin/bash
# AutoPGD evaluation script for fixed MoE experts on CIFAR-100 ResNet
# Run from root folder with: bash bash/evaluate_autopgd_fixed_moe_cifar100_resnet.sh

USE_CLEARML=$1

execute_autopgd () {
  echo "Running AutoPGD evaluation on fixed MoE experts..."
  
  for exp in pgd-adv-train-block ; do
    for routing in global_pooling_linear ; do
      for num_experts in 16 4 2 ; do
        echo "Evaluating: exp=$exp, routing=$routing, num_experts=$num_experts"
        
        if [ "$exp" = "pgd-adv-train-block" ]; then
          run_prefix="pgd-adv-train-block"
        else
          run_prefix="block"
        fi
        
        python eval_fixed_moe.py +experiment=cifar100-resnet18/evaluation_fixed_moe/autopgd $1 \
          model/nn/routing@model.routing_layer_type=$routing \
          moe_layer=model.layer4.1 \
          ++model.num_experts=$num_experts \
          wandb.run_name_prefix=$run_prefix
      done
    done
  done
}

execute_autopgd "+use_clearml=$USE_CLEARML +clearml_queue=\"A100\""