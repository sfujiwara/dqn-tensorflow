#!/usr/bin/env bash

JOB_NAME="metaltile`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`

gcloud ai-platform jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task_metaltile \
  --staging-bucket=gs://${PROJECT_ID}-mlengine \
  --region=us-central1 \
  --config=mlengine-config/config-basic.yaml
