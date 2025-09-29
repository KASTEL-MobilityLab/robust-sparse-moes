#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh


ARGS="trainer.gpus=1"

#python run.py experiment=cifar10-resnet18/attack wandb.checkpoint=
#python run.py experiment=cifar10-resnet18/attack $ARGS wandb.checkpoint_reference=cifar10-resnet18-free-adv
#python run.py experiment=cifar10-resnet18/attack_resnet_block_moe $ARGS wandb.checkpoint_reference=cifar10-resnet18-free-adv-train-block-moe4

python run.py experiment=cifar10-resnet18/attack_resnet_block_moe $ARGS \
       model/nn/routing@model.model.routing_layer_type=conv_global_pooling \
       wandb.checkpoint_reference=cifar10-resnet18-free-adv-train-block-moe4-CGARN
