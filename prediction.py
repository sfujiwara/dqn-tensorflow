# -*- coding: utf-8 -*-

# Default modules
import json
import os

# Additional modules
import numpy as np
import tensorflow as tf

FIELD_SIZE = 8
MODEL_DIR = "gs://cpb100demo1-ml/dqn/dqn20161202170846/model"

with tf.Graph().as_default() as graph:
    saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, "export.meta"))
    inputs = json.loads(tf.get_collection("inputs")[0])
    outputs = json.loads(tf.get_collection("outputs")[0])
    state = graph.get_tensor_by_name(inputs["state"])
    q = graph.get_tensor_by_name(outputs["q"])
    # Create session
    sess = tf.Session()
    saver.restore(sess, os.path.join(MODEL_DIR, "export"))
    for _ in range(10):
        print sess.run(q, feed_dict={state: np.zeros([1, FIELD_SIZE, FIELD_SIZE, 3])})
    # Close session
    sess.close()
