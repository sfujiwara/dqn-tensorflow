# -*- coding: utf-8 -*-

# Additional modules
import numpy as np
import tensorflow as tf

# My modules
import dqn
import repmem
from simulator import chasing

N_INPUTS = 2
N_ITER = 500000
LEARNING_RATE = 1e-3

# Create DQN agent
dqn_agent = dqn.DQN(input_size=N_INPUTS, learning_rate=LEARNING_RATE)
# Create game simulator
chasing_simulator = chasing.ChasingSimulator(field_size=N_INPUTS)
# Create replay memory
replay_memory = repmem.ReplayMemory()

with tf.Session() as sess:
    # Create summary writer and saver
    saver = tf.train.Saver(max_to_keep=1)
    summary_writer = tf.train.SummaryWriter("log", graph=sess.graph)
    # Initializer
    sess.run(tf.initialize_all_variables())
    win_count = 0.
    for i in range(N_ITER):
        if i < 10000:
            action = np.random.randint(5)
        else:
            action = np.argmax(dqn_agent.act(sess, x_t))
            if np.random.uniform() >= 0.8:
                action = np.random.randint(5)
        res = chasing_simulator.input_key(action)
        x_t = res["state_prev"].reshape([1, -1])
        x_t_plus_1 = res["state"].reshape([1, -1])
        terminal = np.atleast_1d(res["terminal"])
        r_t = np.atleast_1d(res["reward"])
        a_t = np.atleast_1d(res["action"])
        replay_memory.store(x_t[0], a_t[0], r_t[0], x_t_plus_1[0], terminal[0])
        mini_batch = replay_memory.sample(size=32)
        train_loss = dqn_agent.update(
            sess,
            mini_batch["s_t"],
            mini_batch["a_t"],
            mini_batch["r_t"],
            mini_batch["s_t_plus_1"],
            mini_batch["terminal"]
        )
        if terminal:
            if r_t > 1 - 1e-5:
                win_count += 1
        win_count += terminal
        if i % 1000 == 0:
            # summary_writer.add_summary(dqn_agent.write_summary(sess), i)
            print("iter: {0} win_rate: {1} reward: {2} loss: {3}".format(i, win_count/(i+1), r_t, np.mean(train_loss)))
