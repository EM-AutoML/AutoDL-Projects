#!/bin/bash
# bash ./scripts-search/search-width-gumbel.sh cifar10 ResNet110 CIFARX 0.57 777
set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for the dataset and the-model-name and the-optimizer and FLOP-ratio and the-random-seed"
  exit 1
fi


dataset=$1
model=$2
optim=$3
expected_FLOP_ratio=$4
rseed=$5

bash ./scripts-search/search-width-cifar.sh ${dataset} ${model} ${optim} 0.1 5 ${expected_FLOP_ratio} ${rseed}
