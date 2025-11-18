#!/bin/bash

set -euC

# $1: task
# $2: gpu number
# $3: save directory

main(){
uv run sample_condition.py \
    --model_config=configs/imagenet_model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=$1 \
    --gpu=$2 \
    --save_dir=$3;
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
