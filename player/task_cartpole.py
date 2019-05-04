# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import gym


parser = argparse.ArgumentParser()
parser.add_argument("--export_dir", type=str, default="sample-models/cartpole-v1")
args, unknown_args = parser.parse_known_args()

MODEL_DIR = args.export_dir


env = gym.make("CartPole-v0")
input_shape = env.observation_space.shape
n_actions = env.action_space.n

frames = []

with tf.Graph().as_default() as g:
    with tf.Session(graph=g) as sess:
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
        # Play game
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = sess.run(q, feed_dict={x_ph: [observation]}).argmax()
            observation, reward, done, info = env.step(action)
        env.close()
