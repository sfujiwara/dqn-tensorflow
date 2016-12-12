# -*- coding: utf-8 -*-

import json
import subprocess
import numpy as np


FIELD_SIZE = 8

field = np.zeros([FIELD_SIZE, FIELD_SIZE, 3])
field[0, 0, 0] = 1
field[2, 1, 1] = 1

data = {
    "key": 0,
    "state": field.tolist()
}

with open('predict_sample.json', 'w') as f:
    json.dump(data, f)

print field[:, :, 0]
print field[:, :, 1]

subprocess.call(
    "gcloud beta ml predict --model=dqn --json-instances=predict_sample.json",
    shell=True
)