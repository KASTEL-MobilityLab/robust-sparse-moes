#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh
USE_CLEARML=$1

execute () {

  #python eval.py +experiment=cifar100-resnet18/evaluation/default $1
  for exp in block; do
    for routing in global_pooling_linear ; do
      for num_experts in 8 ; do
        python eval_fixed_moe.py +experiment=cifar100-resnet18/evaluation_fixed_moe/pgd_adv_train_resnet_${exp}_moe $1 \
          model/nn/routing@model.routing_layer_type=$routing \
          moe_layer=model.layer4.1 \
          ++model.num_experts=$num_experts
      done
      done
  done
#  for exp in block; do
#    for routing in conv_global_pooling ; do
#      for num_experts in 2 4 8 16 32; do
#          python eval_fixed_moe.py +experiment=cifar100-resnet18/evaluation_fixed_moe/resnet_${exp}_moe $1 \
#          model/nn/routing@model.routing_layer_type=$routing \
#          moe_layer=model.layer4.0 \
#          ++model.num_experts=$num_experts
#      done
#    done
#  done
}

execute_free_adv () {

  for exp in block conv; do
    for routing in global_pooling_linear conv_global_pooling; do
      for k in $2; do
        python eval_fixed_moe.py +experiment=cifar100-resnet18/evaluation_fixed_moe/free_adv_train_resnet_${exp}_moe \
          model/nn/routing@model.routing_layer_type=$routing \
          ++model.k=$k $1
      done
      done
  done
}

execute_pgd_adv () {

  for exp in conv block; do
    for routing in conv_global_pooling; do
      for num_experts in 4; do
        python eval_fixed_moe.py +experiment=cifar100-resnet18/evaluation_fixed_moe/pgd_adv_train_resnet_${exp}_moe \
          model/nn/routing@model.routing_layer_type=$routing \
          ++model.num_experts=$num_experts $1
      done
      done
  done
}

# Quick test run to see if any issues come up@
#execute "logger=csv \
#  callbacks=[] \
#  trainer.max_epochs=1\
#  +trainer.limit_val_batches=1 \
#  +trainer.limit_train_batches=1 \
#  +trainer.limit_test_batches=1 \
#  ++datamodule.num_workers=0
#  "


execute "+use_clearml=$USE_CLEARML +clearml_queue="A100""
#execute_free_adv "+use_clearml=true" "1 2 3 4"
#execute "+use_clearml=true"
#execute_pgd_adv #"+use_clearml=true"
