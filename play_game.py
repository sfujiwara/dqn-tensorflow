# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import gym
import numpy as np
from trainer import dqn, chasing


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="CartPole-v1")
parser.add_argument("--env", type=str)
parser.add_argument("--field_size", type=int, default=8)
args, unknown_args = parser.parse_known_args()

ENV_NAME = args.env
CHECKPOINT = args.checkpoint
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

with tf.Graph().as_default() as g:
    agent = dqn.DQN(input_shape=input_shape, n_actions=n_actions, learning_rate=1e-1)


with tf.Session(graph=g) as sess:
    # Play game
    observation = env.reset()
    done = False
    agent.saver.restore(sess, CHECKPOINT)
    while not done:
        env.render()
        action = np.argmax(agent.act(sess, x_t=[observation])[0])
        observation, reward, done, info = env.step(action)
