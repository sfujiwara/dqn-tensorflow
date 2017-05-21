# -*- coding: utf-8 -*-

import argparse
import os
import tensorflow as tf
import gym
import numpy as np
from trainer import dqn


parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int)
args, unknown_args = parser.parse_known_args()

EPISODE = args.episode
ENV_NAME = "CartPole-v1"
CHECKPOINT_DIR = os.path.join("outputs", ENV_NAME, "checkpoints", "model.ckpt-{}".format(EPISODE))

env = gym.make(ENV_NAME)
input_shape = env.observation_space.shape
n_actions = env.action_space.n

with tf.Graph().as_default() as g:
    agent = dqn.DQN(input_shape=input_shape, n_actions=n_actions, learning_rate=1e-1)


with tf.Session(graph=g) as sess:
    # Play game
    observation = env.reset()
    done = False
    agent.saver.restore(sess, CHECKPOINT_DIR)
    while not done:
        env.render()
        action = np.argmax(agent.act(sess, x_t=[observation])[0])
        observation, reward, done, info = env.step(action)
