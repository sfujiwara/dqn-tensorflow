#!/usr/bin/env bash

JOB_NAME="breakout`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_PATH=gs://${PROJECT_ID}-ml/dqn/${JOB_NAME}

gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task_breakout \
  --staging-bucket=gs://${PROJECT_ID}-ml \
  --region=us-central1 \
  --config=config-basic.yaml
