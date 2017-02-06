# -*- coding: utf-8 -*-

# Default modules
import json
import os

# Additional modules
import flask
import numpy as np
import tensorflow as tf

app = flask.Flask(__name__)

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


@app.route('/', methods=["POST"])
def main():
    d = flask.request.json
    print type(d)
    print d
    mat = sess.run(q, feed_dict={state: np.zeros([1, FIELD_SIZE, FIELD_SIZE, 3])}).tolist()
    result = {"predictions": [{"q": i, "key": None} for i in mat]}
    return flask.jsonify(result)

if __name__ == "__main__":
    app.run()
