# -*- coding: utf-8 -*-

# Default modules
import json

# Additional modules
import numpy as np
import tensorflow as tf

with tf.Graph().as_default() as graph:
    saver = tf.train.import_meta_graph("log/model/export.meta")
    inputs = json.loads(tf.get_collection("inputs")[0])
    outputs = json.loads(tf.get_collection("outputs")[0])
    state = graph.get_tensor_by_name(inputs["state"])
    q = graph.get_tensor_by_name(outputs["q"])
    with tf.Session() as sess:
        saver.restore(sess, "log/model/export")
        print sess.run(q, feed_dict={state: np.zeros([1, 84, 84, 3])})
