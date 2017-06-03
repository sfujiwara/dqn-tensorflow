#!/usr/bin/env bash

ENV_NAME="MetalTile-v1"

python -m trainer.task --output_path outputs/${ENV_NAME} \
                       --n_episodes 15000 \
                       --learning_rate 0.01 \
                       --n_updates 1 \
                       --batch_size 100 \
                       --field_size 84 \
                       --env_name ${ENV_NAME}
