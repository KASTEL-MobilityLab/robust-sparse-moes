#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh


execute () {

  python eval.py +experiment=cifar100-resnet18/evaluation/default $1
#  for exp in conv block; do
#    for routing in global_pooling_linear conv_global_pooling; do
#      for k in $2; do
#        python eval.py +experiment=cifar100-resnet18/evaluation/resnet_${exp}_moe \
#          model/nn/routing@model.routing_layer_type=$routing \
#          ++model.k=$k $1
#      done
#    done
#  done
}

execute_free_adv () {

  python eval.py +experiment=cifar100-resnet18/evaluation/free_adv_train $1
  for exp in conv block; do
    for routing in global_pooling_linear conv_global_pooling; do
      for k in $2; do
        python eval.py +experiment=cifar100-resnet18/evaluation/free_adv_train_resnet_${exp}_moe \
          model/nn/routing@model.routing_layer_type=$routing \
          ++model.k=$k $1
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

execute
#execute "+use_clearml=true" "1 3 4"
#execute_free_adv "+use_clearml=true" "1 2 3 4"
#execute_pgd_adv "1 2 3 4"
