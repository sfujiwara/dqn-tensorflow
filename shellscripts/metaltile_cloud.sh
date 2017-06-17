#!/usr/bin/env bash

JOB_NAME="metaltile`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_PATH=gs://${PROJECT_ID}-ml/dqn/${JOB_NAME}
ENV_NAME="MetalTile-v1"

gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket=gs://${PROJECT_ID}-ml \
  --region=us-central1 \
  --config=config-gpu.yaml \
  -- \
  --output_path="${TRAIN_PATH}/outputs" \
  --n_episodes=50000 \
  --learning_rate=0.0001 \
  --n_updates=50 \
  --batch_size=32 \
  --field_size=8 \
  --random_action_decay 0.9999 \
  --replay_memory_size 5000 \
  --env_name=${ENV_NAME}
