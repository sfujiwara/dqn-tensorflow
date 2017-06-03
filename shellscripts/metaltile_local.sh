#!/usr/bin/env bash

ENV_NAME="MetalTile-v1"

python -m trainer.task --output_path outputs/${ENV_NAME} \
                       --n_episodes 5000 \
                       --learning_rate 0.001 \
                       --n_updates 5 \
                       --batch_size 50 \
                       --field_size 84 \
                       --env_name ${ENV_NAME}
