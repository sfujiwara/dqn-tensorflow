#!/usr/bin/env bash

JOB_NAME="metaltile`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_PATH=gs://${PROJECT_ID}-ml/dqn/${JOB_NAME}
ENV_NAME="MetalTile-v1"

gcloud ai-platform jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task_metaltile \
  --staging-bucket=gs://${PROJECT_ID}-mlengine \
  --region=us-central1 \
  --config=mlengine-config/config-basic.yaml
