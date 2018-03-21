#!/usr/bin/env bash

PROJECT_ID="kaggle-playground"
INSTANCE_NAME="dqn-breakout-`date '+%Y%m%d%H%M%S'`"

# Create Compute Engine instance
gcloud beta compute --project ${PROJECT_ID} instances create ${INSTANCE_NAME} \
  --zone "us-central1-b" \
  --machine-type "n1-highmem-8" \
  --subnet "default" \
  --maintenance-policy "MIGRATE" \
  --service-account "701154693276-compute@developer.gserviceaccount.com" \
  --scopes "https://www.googleapis.com/auth/cloud-platform" \
  --min-cpu-platform "Automatic" \
  --tags "http-server","https-server" \
  --image "ubuntu-1604-xenial-v20180306" \
  --image-project "ubuntu-os-cloud" \
  --boot-disk-size "20" \
  --boot-disk-type "pd-standard" \
  --boot-disk-device-name ${INSTANCE_NAME} \
  --metadata-from-file startup-script=startup.sh
