# Deep Q Networks with TensorFlow

## Training on Local

```sh
python -m trainer.task --output_path=log
```

## Training on Cloud Machine Learning

```sh
JOB_NAME="dqn`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_BUCKET=gs://${PROJECT_ID}-ml
TRAIN_PATH=${TRAIN_BUCKET}/dqn/${JOB_NAME}
gsutil rm -rf ${TRAIN_PATH}
gsutil cp .dummy ${TRAIN_PATH}/model/

gcloud beta ml jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-ml" \
  --region=us-central1 \
  --config=config.yaml \
  -- \
  --output_path="${TRAIN_PATH}"
```

## Prediction on Cloud Machine Learning

```
gcloud beta ml predict --model=dqn --instances=predict_sample.json
```

```yaml
predictions:
- key: 0
  q:
  - 0.711304
  - 0.766368
  - 0.757209
  - 0.75318
  - 0.737671
```