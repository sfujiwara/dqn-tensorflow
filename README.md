# Deep Q-Networks with TensorFlow

## Training on Local

```sh
python -m trainer.task --output_path=log \
                       --max_epochs=5000 \
                       --learning_rate=0.001 \
                       --n_inputs=2
```

## Training on Cloud Machine Learning

```sh
JOB_NAME="dqn`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_BUCKET=gs://${PROJECT_ID}-ml
TRAIN_PATH=${TRAIN_BUCKET}/dqn/${JOB_NAME}
gsutil cp .dummy ${TRAIN_PATH}/model/

gcloud beta ml jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-ml" \
  --region=us-central1 \
  --config=config.yaml \
  -- \
  --output_path="${TRAIN_PATH}"
  --max_epochs=50000
  --learning_rate=0.001
```

## Prediction on Cloud Machine Learning

### using `gcloud beta ml predict`

```
gcloud beta ml predict --model=dqn --json-instances=predict_sample.json
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

### using `curl`

```
ACCESS_TOKEN=`gcloud auth print-access-token`
curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer ${ACCESS_TOKEN}" https://ml.googleapis.com/v1beta1/projects/cpb100demo1/models/dqn:predict -d @predict_sample_curl.json
```

```json
{"predictions": [{"q": [0.4523559808731079, 0.38499385118484497, 0.26314204931259155, 0.6228029131889343, 0.5784728527069092], "key": 0}]} 
```
