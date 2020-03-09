#!/bin/bash 

export CUDA_VISIBLE_DEVICES=0
seed=66


bash ./scripts-search/search-width-gumbel.sh cifar10 ResNet32 CIFARX 0.57 $seed &
bash ./scripts-search/search-width-gumbel.sh cifar100 ResNet32 CIFARX 0.57 $seed &
fsdal;kfjsdl;
wait