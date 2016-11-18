# Deep Q Networks with TensorFlow

## Training on Local

```sh
python -m trainer.task.py
```

## Training on Cloud Machine Learning

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