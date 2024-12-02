#!/bin/bash

# gpt2
# trained-t5
# google-t5/t5-3b
# gpt-4o

model="gpt-4o"
gpu=2
version="base"

python -u \
    analyze_result.py\
    --model $model \
    --gpu $gpu \
    --version $version \
    >analyze_result_$version.log 2>&1 &