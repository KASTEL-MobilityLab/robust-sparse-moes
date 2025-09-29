#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

execute () {

  #python run.py experiment=cifar10-resnet18/default $1
  #python run.py experiment=cifar10-resnet18/resnet_block_moe $1
  python run.py experiment=cifar10-resnet18/resnet_block_moe model/nn/routing@model.model.routing_layer_type=conv_global_pooling $1
  #python run.py experiment=cifar10-resnet18/free_adv_train $1
  #python run.py experiment=cifar10-resnet18/free_adv_train_resnet_block_moe $1
  python run.py experiment=cifar10-resnet18/free_adv_train_resnet_block_moe model/nn/routing@model.model.routing_layer_type=conv_global_pooling $1

}

# Quick test run to see if any issues come up@
execute "logger=csv \
  callbacks=[] \
  trainer.max_epochs=1\
  +trainer.limit_val_batches=1 \
  +trainer.limit_train_batches=1 \
  +trainer.limit_test_batches=1 \
  trainer.gpus=1
  "


execute "trainer.max_epochs=100 trainer.gpus=1"
