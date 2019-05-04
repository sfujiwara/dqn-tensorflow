# Deep Q-Networks with TensorFlow

## Requirements

- Pipenv
- pyenv
- gcloud SDK (optional)

## Setting Up

```bash
pipenv install
```

## Training on Local

```sh
pipenv run train-cartpole
```

```sh
pipenv run train-metaltile
```

## Training on Google Cloud Machine Learning Engine

```sh
pipenv run train-metaltile-cloud
```

## Play Game by Trained Model

```sh
# Specify export directory for example:
EXPORT_DIR="sample-models/CartPole-v1/models/episode-4970"

# Specify the name of environment for example:
ENV_NAME="CartPole-v1"

python -m utils.play_game --env ${ENV_NAME} --export_dir ${EXPORT_DIR}
```

### using `curl`

```
ACCESS_TOKEN=`gcloud auth print-access-token`
curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer ${ACCESS_TOKEN}" https://ml.googleapis.com/v1beta1/projects/cpb100demo1/models/dqn:predict -d @sampledata/sample_curl.json
```

```json
{"predictions": [{"q": [0.4523559808731079, 0.38499385118484497, 0.26314204931259155, 0.6228029131889343, 0.5784728527069092], "key": 0}]}
```

## DQN Server

```
python dqn_server.py --model ${PATH_TO_SAVED_MODEL}
```

```
curl -X POST -H "Content-Type: application/json" localhost:5000 -d @sampledata/sample_curl.json
```
