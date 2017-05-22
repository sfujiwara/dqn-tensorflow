# -*- coding: utf-8 -*-

# Default modules
import argparse
import json
import os

import numpy as np
import tensorflow as tf
import gym

from trainer import dqn, repmem
from trainer import chasing


# Set log level
tf.logging.set_verbosity(tf.logging.DEBUG)

# Print TensorFlow version
tf.logging.info(tf.__version__)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)
parser.add_argument("--n_episodes", type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--env_name", type=str)
# parser.add_argument("--random_action_prob", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--n_updates", type=int, default=10)

args, unknown_args = parser.parse_known_args()
tf.logging.info("known args: {}".format(args))

# Set constant values
N_EPISODES = args.n_episodes
N_RANDOM_ACTION = 20  # int(N_EPISODES / 10) + 1
LEARNING_RATE = args.learning_rate
ENV_NAME = args.env_name
# RANDOM_ACTION_PROB = args.random_action_prob
OUTPUT_PATH = args.output_path
BATCH_SIZE = args.batch_size
N_UPDATES = args.n_updates

# Get environment variable for Cloud ML
tf_conf = json.loads(os.environ.get("TF_CONFIG", "{}"))
# For local
if not tf_conf:
    tf_conf = {
      "cluster": {"master": ["localhost:2222"]},
      "task": {"index": 0, "type": "master"}
    }
tf.logging.debug("TF_CONF: {}".format(json.dumps(tf_conf, indent=2)))

# Cluster setting for cloud
cluster = tf_conf.get("cluster", None)

server = tf.train.Server(
    cluster,
    job_name=tf_conf["task"]["type"],
    task_index=tf_conf["task"]["index"]
)

# Parameter server
if tf_conf["task"]["type"] == "ps":
    server.join()

# Master and workers
device_fn = tf.train.replica_device_setter(
    cluster=tf.train.ClusterSpec(cluster=cluster),
    worker_device="/job:{0}/task:{1}".format(tf_conf["task"]["type"], tf_conf["task"]["index"]),
)

# Logging
tf.logging.debug("/job:{0}/task:{1} build graph".format(tf_conf["task"]["type"], tf_conf["task"]["index"]))

# Create game simulator
# env = chasing.ChasingSimulator(field_size=N_INPUTS)
env = gym.make(ENV_NAME)
input_shape = env.observation_space.shape
n_actions = env.action_space.n

tf.logging.info("Input Shape: {}".format(input_shape))
tf.logging.info("The Number of Actions: {}".format(n_actions))

# Create replay memory
replay_memory = repmem.ReplayMemory(memory_size=2000)

# Build graph
with tf.Graph().as_default() as graph:

    with tf.device(device_fn):
        # Create DQN agent
        dqn_agent = dqn.DQN(input_shape=input_shape, learning_rate=LEARNING_RATE, n_actions=n_actions)
        global_step = tf.Variable(0, trainable=False, name="global_step")
        increment_global_step_op = global_step.assign_add(1)
        if tf_conf["task"]["type"] == "master":
            summary_writer = tf.summary.FileWriter(os.path.join(OUTPUT_PATH, "summaries"), graph=graph)

    with tf.train.MonitoredTrainingSession(
        master=server.target,
        is_chief=(tf_conf["task"]["type"] == "master"),
        checkpoint_dir=os.path.join(OUTPUT_PATH, "checkpoints"),
        scaffold=None,
        hooks=None,
        chief_only_hooks=None,
        save_checkpoint_secs=60,
        save_summaries_steps=None,
        # save_summaries_secs=None,
        config=None,
        # stop_grace_period_secs=120,
    ) as mon_sess:
        random_action_prob = 1.0
        for i in range(N_EPISODES):
            # Play a new game
            previous_observation = env.reset()
            done = False
            total_reward = 0
            for j in range(1000):
                # Act at random with a fixed probability
                if np.random.rand() <= random_action_prob:
                    action = np.random.randint(n_actions)
                # Act following the policy on the other games
                else:
                    q = dqn_agent.act(mon_sess, np.array([previous_observation]))
                    action = q.argmax()
                # Receive the results from the game simulator
                observation, reward, done, info = env.step(action)
                reward = reward if not done else -10
                total_reward += reward
                # Store the experience
                replay_memory.store(previous_observation, action, reward, observation, done)
                previous_observation = observation
                # Update the policy
                for _ in range(N_UPDATES):
                    mini_batch = replay_memory.sample(size=BATCH_SIZE)
                    train_loss = dqn_agent.update(
                        mon_sess, mini_batch[0], mini_batch[1], mini_batch[2], mini_batch[3], mini_batch[4]
                    )
                if done:
                    break
            tf.logging.info(
                "Episode: {0} Total Reward: {1} Training Loss: {2} Random Action Probability: {3}".format(
                mon_sess.run(global_step), total_reward, np.mean(train_loss), random_action_prob)
            )
            env.reset()
            mon_sess.run(increment_global_step_op)
            random_action_prob = max(random_action_prob * 0.999, 0.05)
