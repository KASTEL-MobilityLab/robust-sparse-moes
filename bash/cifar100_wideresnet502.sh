#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

K=$1

execute_entropy () {

  python run.py experiment=cifar100-wideresnet502/default $1 +wandb.unique_name='${name}'
  for exp in conv block; do
    for routing in global_pooling_linear conv_global_pooling; do
      for k in 1 2 3 4; do
        python3 run.py experiment=cifar100-wideresnet502/resnet_${exp}_moe \
          model/nn/routing@model.model.routing_layer_type=$routing \
          model.model.k=$k $1 \
          model.model.num_experts=4 \
	  model.model.balancing_loss_type="entropy" \
	  model.model.balancing_loss=0.5
      done
    done
  done

}


execute_switch () {

  #python run.py experiment=cifar100-resnet18/default $1 +wandb.unique_name='${name}'
  for exp in conv block; do
    for routing in global_pooling_linear conv_global_pooling; do
      for k in 1 2 3 4; do
        python3 run.py experiment=cifar100-wideresnet502/wideresnet_${exp}_moe \
          model/nn/routing@model.model.routing_layer_type=$routing \
          model.model.k=$k $1 \
	  model.model.num_experts=4 \
	  model.model.balancing_loss_type="switch" \
	  model.model.balancing_loss=0.01
      done
    done
  done
}

execute_pgd_adv_switch () {

  #python run.py experiment=cifar100-wideresnet502/pgd_adv_train $1 +wandb.unique_name='${name}'
  #python run.py --multirun \
  #  experiment=cifar100-resnet18/resnet_block_moe,cifar100-resnet18/resnet_conv_moe \
  #  model/nn/routing@model.model.routing_layer_type=global_pooling_linear,conv_global_pooling \
  #  model.model.k=1,2,3,4 $1

  for exp in conv block; do
    for routing in global_pooling_linear conv_global_pooling; do
      for k in 1 2 3 4; do
        python run.py experiment=cifar100-wideresnet502/pgd_adv_train_resnet_${exp}_moe \
          model/nn/routing@model.model.routing_layer_type=$routing \
          model.model.k=$k $1 \
          model.model.num_experts=4 \
          model.model.balancing_loss_type="switch" \
          model.model.balancing_loss=0.01
      done
    done
  done
}

execute_free_adv () {

  for exp in block; do
    for routing in conv_global_pooling; do
      for num_experts in 4; do
        python run.py experiment=cifar100-resnet18/free_adv_train_resnet_${exp}_moe \
          model/nn/routing@model.model.routing_layer_type=$routing \
          model.model.k=1 $1 +wandb.unique_name='${name}' \
          model.model.num_experts=$num_experts
      done
    done
  done
}
execute_fast_adv () {

  python run.py experiment=cifar100-resnet18/fast_adv_train $1 +wandb.unique_name='${name}'
  for exp in block; do
    for routing in conv_global_pooling; do
      for num_experts in 4; do
        python run.py experiment=cifar100-resnet18/fast_adv_train_resnet_${exp}_moe \
          model/nn/routing@model.model.routing_layer_type=$routing \
          model.model.k=1 $1 +wandb.unique_name='${name}' \
          model.model.num_experts=$num_experts
      done
    done
  done
}


DEFAULT_DEBUG_PARAMS="logger=csv \
  callbacks=[] \
  trainer.max_epochs=1\
  +trainer.limit_val_batches=1 \
  +trainer.limit_train_batches=1 \
  +trainer.limit_test_batches=1 \
  +use_clearml=false \
  trainer.gpus=1"

# Quick test run to see if any issues come up@
#execute "$DEFAULT_DEBUG_PARAMS" "1,2,3,4"
#execute_entropy "trainer.max_epochs=200 trainer.gpus=1 +trainer.precision=16 +use_clearml=false"
#execute_switch "trainer.max_epochs=200 trainer.gpus=1 +trainer.precision=16 +use_clearml=false"

#execute_free_adv "$DEFAULT_DEBUG_PARAMS" "1,2,3,4"
#execute_free_adv "trainer.max_epochs=50 trainer.gpus=1 +use_clearml=true"
#execute_fast_adv "trainer.max_epochs=200 trainer.gpus=1 +use_clearml=true"

#execute_pgd_adv "$DEFAULT_DEBUG_PARAMS"
execute_pgd_adv_switch "trainer.max_epochs=200 trainer.gpus=1 +trainer.precision=16 +use_clearml=false"
#execute_pgd_adv "trainer.max_epochs=200 trainer.gpus=1 +use_clearml=true ++clearml_queue=rtx3090"
#execute_pgd_adv "trainer.max_epochs=200 trainer.gpus=1 +use_clearml=true"
