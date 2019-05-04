#!/usr/bin/env bash

#!/usr/bin/env bash

ENV_NAME="MountainCar-v0"

python -m trainer.task --output_path outputs/${ENV_NAME} \
                       --n_episodes 15000 \
                       --learning_rate 0.001 \
                       --n_updates 2 \
                       --batch_size 100 \
                       --random_action_decay 0.9999 \
                       --env_name ${ENV_NAME}
