#!/bin/bash


dataset="CounterCoTQA"
prompt="explanation"
version="base"
gpu=1
model='trained-t5'

output_prefix="outputs/${dataset}/$model"
mkdir -p $output_prefix

python -u \
    contrastive_decoding_rationalization.py \
    --output_prefix $output_prefix \
    --model $model\
    --dataset $dataset \
    --prompt $prompt \
    --version $version\
    --gpu $gpu\
    >${output_prefix}/rationalization.${version}.log 2>&1 &
