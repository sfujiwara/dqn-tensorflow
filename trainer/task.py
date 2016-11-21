# -*- coding: utf-8 -*-

# Default modules
import argparse
import json
import os

import numpy as np
import tensorflow as tf

from trainer import dqn, repmem
from trainer import chasing


# Set log level
tf.logging.set_verbosity(tf.logging.DEBUG)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--learning_rate", type=float)
args, unknown_args = parser.parse_known_args()
tf.logging.info("known args: {}".format(args))

# Set constant values
MAX_EPOCHS = args.max_epochs
N_INPUTS = 3
N_RANDOM_ACTION = 5000
LEARNING_RATE = args.learning_rate
N_ACTIONS = 5

# Get environment variable for Cloud ML
tf_conf = json.loads(os.environ.get("TF_CONFIG", "{}"))
# For local
if not tf_conf:
    tf_conf = {
      "cluster": {"master": ["localhost:2222"]},
      "task": {"index": 0, "type": "master"}
    }
tf.logging.debug("TF_CONF: {}".format(json.dumps(tf_conf)))

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
else:
    device_fn = tf.train.replica_device_setter(
        cluster=tf.train.ClusterSpec(cluster=cluster),
        worker_device="/job:{0}/task:{1}".format(tf_conf["task"]["type"], tf_conf["task"]["index"]),
    )

    # Logging
    tf.logging.debug("/job:{0}/task:{1} build graph".format(tf_conf["task"]["type"], tf_conf["task"]["index"]))

    # Build graph
    with tf.Graph().as_default() as graph:
        with tf.device(device_fn):
            # Create DQN agent
            dqn_agent = dqn.DQN(input_size=N_INPUTS, learning_rate=LEARNING_RATE, n_actions=N_ACTIONS)
            global_step = tf.Variable(0, trainable=False, name="global_step")
            win_count = tf.Variable(0, trainable=False, name="win_count")
            increment_win_count_op = win_count.assign_add(1)
            init_op = tf.initialize_all_variables()
            # Create saver
            saver = tf.train.Saver(max_to_keep=10)

    # Create game simulator
    game_simulator = chasing.ChasingSimulator(field_size=N_INPUTS)
    # Create replay memory
    replay_memory = repmem.ReplayMemory()

    sv = tf.train.Supervisor(
        graph=graph,
        is_chief=(tf_conf["task"]["type"] == "master"),
        logdir=args.output_path,
        init_op=init_op,
        global_step=global_step,
        summary_op=None
    )

    with sv.managed_session(server.target) as sess:
        # Create summary writer
        summary_writer = tf.train.SummaryWriter(args.output_path, graph=sess.graph)
        # Initializer
        sess.run(init_op)
        win_count = 0
        for i in range(MAX_EPOCHS):
            # Play a new game
            while not game_simulator.terminal:
                # Act at random on the first few games
                if i < N_RANDOM_ACTION:
                    action = np.random.randint(N_ACTIONS)
                # Act at random with a fixed probability
                elif np.random.uniform() >= 0.9:
                    action = np.random.randint(N_ACTIONS)
                # Act following the policy on the other games
                else:
                    action = np.argmax(dqn_agent.act(sess, np.array([s_t])))
                # Act on the game simulator
                res = game_simulator.input_key(action)
                # Receive the results from the game simulator
                s_t = res["s_t"]
                s_t_plus_1 = res["s_t_plus_1"]
                terminal = res["terminal"]
                r_t = res["r_t"]
                a_t = res["a_t"]
                # Store the experience
                replay_memory.store(s_t, a_t, r_t, s_t_plus_1, terminal)
                # Update the policy
                mini_batch = replay_memory.sample(size=32)
                train_loss = dqn_agent.update(
                    sess,
                    mini_batch["s_t"],
                    mini_batch["a_t"],
                    mini_batch["r_t"],
                    mini_batch["s_t_plus_1"],
                    mini_batch["terminal"]
                )
            tf.logging.info("epoch: {0} win_rate: {1} reward: {2} loss: {3}".format(
                i, win_count/(i+1e-5), r_t, np.mean(train_loss))
            )
            if r_t > 0.5:
                sess.run(increment_win_count_op)
                win_count += 1
            game_simulator.init_game()
        # Save model
        dqn_agent.save_model(sess, args.output_path)
        sv.stop()
