#!/bin/bash

dataset="CounterCoTQA"
version="base"
model_name="gpt2"
teacher_model="gpt-neox-20b"
max_enc_length=128
max_dec_length=128
train_batch_size=8
eval_batch_size=32
grad_step=1
learning_rate=1e-5
warmup_ratio=0.06
weight_decay=0
num_epoch=3
num_epoch_early_stopping=1


counterfactual_alpha=0.5

gpu=2

save_dir="checkpoints/${dataset}/$version/counterfactual${counterfactual_alpha}_${model_name}_bs${train_batch_size}_gs${grad_step}_lr${learning_rate}_wd${weight_decay}_e${num_epoch}"
mkdir -p $save_dir

python -u \
    main.py \
    --num_epoch_early_stopping $num_epoch_early_stopping \
    --add_task_prefix \
    --use_wandb\
    --teacher_model $teacher_model\
    --wandb_project mikd\
    --counterfactual_alpha $counterfactual_alpha \
    --dataset $dataset \
    --save_dir $save_dir \
    --model_name $model_name \
    --max_enc_length $max_enc_length \
    --max_dec_length $max_dec_length \
    --train_batch_size $train_batch_size \
    --eval_batch_size $eval_batch_size \
    --grad_step $grad_step \
    --learning_rate $learning_rate \
    --warmup_ratio $warmup_ratio \
    --weight_decay $weight_decay \
    --num_epoch $num_epoch \
    --version $version\
    --gpu $gpu\
    > ${save_dir}/debug.log 2>&1 &

