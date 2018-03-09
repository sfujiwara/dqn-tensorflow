#!/usr/bin/env bash

ENV_NAME="Breakout"

python -m trainer.task --output_path outputs/${ENV_NAME} \
                       --n_episodes 5000 \
                       --learning_rate 0.0005 \
                       --n_updates 10 \
                       --batch_size 32 \
                       --random_action_decay 0.999 \
                       --env_name ${ENV_NAME}
