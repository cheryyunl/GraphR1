#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/code/scenes/data_extended@train \
    data.val_files=/code/scenes/data_extended@validation \
    data.format_prompt=./examples/format_prompt/dapo_graph.jinja \
    data.mini_rollout_batch_size=32 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.reward.reward_function=./examples/reward_function/dapo_graph.py:compute_score \
    worker.reward.reward_function_kwargs='{"max_response_length":2048,"overlong_buffer_length":512,"overlong_penalty_factor":0.5,"format_weight":0.2}' \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    trainer.experiment_name=qwen2_5_vl_7b_graph_dapo \
    trainer.n_gpus_per_node=8
