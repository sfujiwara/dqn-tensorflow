#!/usr/bin/env bash

ENV_NAME="MetalTile-v1"

python -m trainer.task --output_path outputs/${ENV_NAME} \
                       --n_episodes 15000 \
                       --learning_rate 0.005 \
                       --n_updates 10 \
                       --batch_size 32 \
                       --field_size 8 \
                       --random_action_decay 0.9999 \
                       --replay_memory_size 10000 \
                       --env_name ${ENV_NAME}
