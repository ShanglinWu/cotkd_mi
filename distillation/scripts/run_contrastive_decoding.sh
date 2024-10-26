#!/bin/bash

dataset="CounterCoTQA"
prompt="explanation"
version="distractor"
gpu=2

output_prefix="outputs/${dataset}/"
mkdir -p $output_prefix

python -u \
    contrastive_decoding_rationalization.py \
    --output_prefix $output_prefix \
    --dataset $dataset \
    --prompt $prompt \
    --version $version\
    --gpu $gpu\
    >${output_prefix}/rationalization.${version}.log 2>&1 &
