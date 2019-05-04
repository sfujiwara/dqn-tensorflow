# -*- coding: utf-8 -*-

# Default modules
import argparse

# Additional modules
import flask
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args, unknown_args = parser.parse_known_args()

app = flask.Flask(__name__)

FIELD_SIZE = 8
MODEL_DIR = args.model

with tf.Graph().as_default() as g:
    sess = tf.Session()
    meta_graph = tf.saved_model.loader.load(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        export_dir=MODEL_DIR
    )
    model_signature = meta_graph.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_signature = model_signature.inputs
    output_signature = model_signature.outputs
    # Get names of input and output tensors
    input_tensor_name = input_signature["state"].name
    output_tensor_name = output_signature["q"].name
    # Get input and output tensors
    x_ph = sess.graph.get_tensor_by_name(input_tensor_name)
    q = sess.graph.get_tensor_by_name(output_tensor_name)


@app.route('/', methods=["GET", "POST"])
def main():
    content = flask.request.get_json(force=True)
    result = sess.run(q, feed_dict={x_ph: [content["instances"][0]["state"]]}).tolist()
    return flask.jsonify({"predictions": [{"q": result[0], "key": None}]})

if __name__ == "__main__":
    app.run(debug=False)
