# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import dqn
import emulator

dqn_agent = dqn.DQN()
simple_emulator = emulator.SampleEmulator()

x_t = np.random.uniform(size=[32, 84, 84, 4])
x_t_plus_1 = np.random.uniform(size=[32, 84, 84, 4])
r_t = np.random.uniform(size=32)
terminate = np.ones(32, dtype=bool)


with tf.Session() as sess:
    tf.train.SummaryWriter("log", graph=sess.graph)
    sess.run(tf.initialize_all_variables())
    # Generate data using emulator
    for _ in range(100):
        a_t = dqn_agent.act(sess, x_t)
        train_loss = dqn_agent.update(sess, x_t, x_t_plus_1, r_t, terminate)
        print np.mean(train_loss)
