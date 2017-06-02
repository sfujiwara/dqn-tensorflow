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
  --config=config.yaml \
  -- \
  --output_path="${TRAIN_PATH}/outputs" \
  --n_episodes=10000 \
  --learning_rate=0.001 \
  --n_updates=10 \
  --batch_size=50 \
  --field_size=84 \
  --env_name=${ENV_NAME}