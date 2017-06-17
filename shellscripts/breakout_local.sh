#!/usr/bin/env bash

ENV_NAME="Breakout-v0"

python -m trainer.task --output_path outputs/${ENV_NAME} \
                       --n_episodes 5000 \
                       --learning_rate 0.001 \
                       --n_updates 1 \
                       --batch_size 50 \
                       --random_action_decay 0.999 \
                       --env_name ${ENV_NAME}
