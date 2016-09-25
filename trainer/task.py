# -*- coding: utf-8 -*-

# Additional modules
import numpy as np
import tensorflow as tf

# My modules
import dqn
from simulator import chasing

ACTIONS = ["right", "left", "up", "down", "do_nothing"]
N_INPUTS = 8

dqn_agent = dqn.DQN(n_channels=1, input_size=N_INPUTS)
chasing_simulator = chasing.ChasingSimulator(field_size=N_INPUTS)

with tf.Session() as sess:
    tf.train.SummaryWriter("log", graph=sess.graph)
    sess.run(tf.initialize_all_variables())
    for i in range(10000):
        if i == 0:
            a_t = 0
        else:
            a_t = np.argmax(dqn_agent.act(sess, x_t))
        if np.random.randint(100) >= 79:
            a_t = np.random.randint(5)
        res = chasing_simulator.input_key(ACTIONS[a_t])
        x_t = res["state_prev"].reshape([1, N_INPUTS, N_INPUTS, 1])
        x_t_plus_1 = res["state"].reshape([1, N_INPUTS, N_INPUTS, 1])
        terminal = res["terminal"]
        r_t = res["reward"]
        train_loss = dqn_agent.update(sess, x_t, x_t_plus_1, r_t, terminal)
        # print("action: {}".format(np.argmax(a_t)))
        print np.mean(train_loss), r_t, a_t, ACTIONS[a_t]
        if terminal:
            print("========== GAME SET!! ==========")
