# -*- coding: utf-8 -*-

import json
import numpy as np

FIELD_SIZE = 84

data = {
    "key": 0,
    "state": np.zeros([FIELD_SIZE, FIELD_SIZE, 3]).tolist()
}

with open('predict_sample.json', 'w') as f:
    json.dump(data, f)
