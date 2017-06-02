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
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--n_updates", type=int, default=10)
parser.add_argument("--field_size", type=int, default=16)

args, unknown_args = parser.parse_known_args()
tf.logging.info("known args: {}".format(args))

# Set constant values
N_EPISODES = args.n_episodes
LEARNING_RATE = args.learning_rate
ENV_NAME = args.env_name
OUTPUT_PATH = args.output_path
BATCH_SIZE = args.batch_size
N_UPDATES = args.n_updates
FIELD_SIZE = args.field_size

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
if ENV_NAME == "Chasing-v1":
    # Use my environment
    env = chasing.ChasingEnv(field_size=FIELD_SIZE)
else:
    # Use OpenAI Gym environment
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
        # Only master build summary graph for TensorBoard
        if tf_conf["task"]["type"] == "master":
            reward_summary_ph = tf.placeholder(tf.float32)
            summary_writer = tf.summary.FileWriter(os.path.join(OUTPUT_PATH, "summaries"), graph=graph)
            tf.summary.scalar(name="reward", tensor=reward_summary_ph)
            summary_op = tf.summary.merge_all()

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
        while True:
            random_action_prob = max(0.999 ** mon_sess.run(global_step), 0.05)
            # Play a new game
            previous_observation = env.reset()
            done = False
            total_reward = 0
            while not done:
                # Act at random with a fixed probability
                if np.random.rand() <= random_action_prob:
                    action = np.random.randint(n_actions)
                # Act following the policy on the other games
                else:
                    q = dqn_agent.act(mon_sess, np.array([previous_observation]))
                    action = q.argmax()
                # Receive the results from the game simulator
                observation, reward, done, info = env.step(action)
                # reward = reward if not done else -10
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
            tf.logging.info(
                "Episode: {0} Total Reward: {1} Training Loss: {2} Random Action Probability: {3}".format(
                mon_sess.run(global_step), total_reward, np.mean(train_loss), random_action_prob)
            )
            env.reset()
            mon_sess.run(increment_global_step_op)
            # Only master write summary
            if tf_conf["task"]["type"] == "master":
                summary_str = mon_sess.run(summary_op, {reward_summary_ph: total_reward})
                summary_writer.add_summary(summary_str, mon_sess.run(global_step))
                summary_writer.flush()
            # Termination
            if mon_sess.run(global_step) >= N_EPISODES:
                break

        if tf_conf["task"]["type"] == "master":
            dqn_agent.save_model(dir=OUTPUT_PATH)
