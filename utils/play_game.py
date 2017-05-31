# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import gym
from trainer import chasing


parser = argparse.ArgumentParser()
parser.add_argument("--export_dir", type=str)
parser.add_argument("--env", type=str)
parser.add_argument("--field_size", type=int, default=8)
args, unknown_args = parser.parse_known_args()

ENV_NAME = args.env
MODEL_DIR = args.export_dir
FIELD_SIZE = args.field_size

# Create game simulator
if ENV_NAME == "Chasing-v1":
    # Use my environment
    env = chasing.ChasingEnv(field_size=8)
else:
    # Use OpenAI Gym environment
    env = gym.make(ENV_NAME)
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
