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

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)
parser.add_argument("--n_episodes", type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--env_name", type=str)
parser.add_argument("--random_action_prob", type=float, default=0.1)
args, unknown_args = parser.parse_known_args()
tf.logging.info("known args: {}".format(args))

# Set constant values
N_EPISODES = args.n_episodes
N_RANDOM_ACTION = int(N_EPISODES / 20) + 1
LEARNING_RATE = args.learning_rate
ENV_NAME = args.env_name
RANDOM_ACTION_PROB = args.random_action_prob
OUTPUT_PATH = args.output_path

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
replay_memory = repmem.ReplayMemory()

# Build graph
with tf.Graph().as_default() as graph:

    with tf.device(device_fn):
        # Create DQN agent
        dqn_agent = dqn.DQN(input_shape=input_shape, learning_rate=LEARNING_RATE, n_actions=n_actions)
        global_step = tf.Variable(0, trainable=False, name="global_step")
        increment_global_step_op = global_step.assign_add(1)
        # init_op = tf.global_variables_initializer()
    summary_writer = tf.summary.FileWriter(os.path.join(OUTPUT_PATH, "summaries"), graph=graph)

    with tf.train.MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=os.path.join(OUTPUT_PATH, "checkpoints"),
        scaffold=None,
        hooks=None,
        chief_only_hooks=None,
        save_checkpoint_secs=600,
        save_summaries_steps=None,
        save_summaries_secs=None,
        config=None,
        stop_grace_period_secs=120,
    ) as mon_sess:
        for i in range(N_EPISODES):
            # Play a new game
            previous_observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                # Act at random on the first few games
                if i < N_RANDOM_ACTION:
                    action = np.random.randint(n_actions)
                # Act at random with a fixed probability
                elif np.random.uniform() <= RANDOM_ACTION_PROB:
                    action = np.random.randint(n_actions)
                    # action = 0
                # Act following the policy on the other games
                else:
                    action = np.argmax(dqn_agent.act(mon_sess, np.array([previous_observation])))
                    print action,
                # Receive the results from the game simulator
                observation, reward, done, info = env.step(action)
                total_reward += reward
                # Store the experience
                replay_memory.store(previous_observation, action, reward, observation, done)
                previous_observation = observation
                # Update the policy
                for _ in range(10):
                    mini_batch = replay_memory.sample(size=100)
                    train_loss = dqn_agent.update(
                        mon_sess,
                        mini_batch["s_t"],
                        mini_batch["a_t"],
                        mini_batch["r_t"],
                        mini_batch["s_t_plus_1"],
                        mini_batch["terminal"]
                    )
            tf.logging.info("Episode: {0} Total Reward: {1} Training Loss: {2}".format(
                i, total_reward, np.mean(train_loss))
            )
            env.reset()
            mon_sess.run(increment_global_step_op)
        # Only master save model
        if tf_conf["task"]["type"] == "master":
            tf.logging.debug("save model to {}".format(args.output_path))
            dqn_agent.save_model(mon_sess, args.output_path)
