# -*- coding: utf-8 -*-

# Additional modules
import numpy as np
import tensorflow as tf

from trainer import dqn, repmem
from trainer.simulator import chasing

N_INPUTS = 5
N_EPOCH = 500
LEARNING_RATE = 1e-4
N_ACTIONS = 5
N_KERNEL = 2

# Create DQN agent
dqn_agent = dqn.DQN(input_size=N_INPUTS, learning_rate=LEARNING_RATE, n_actions=N_ACTIONS)

# Create saver
saver = tf.train.Saver(max_to_keep=1)

# Create game simulator
# chasing_simulator = chasing.ChasingSimulator(field_size=N_INPUTS)
game_simulator = chasing.ChasingSimulator(field_size=N_INPUTS)

# Create replay memory
replay_memory = repmem.ReplayMemory()

with tf.Session() as sess:
    # Create summary writer
    summary_writer = tf.train.SummaryWriter("log", graph=sess.graph)
    # Initializer
    sess.run(tf.initialize_all_variables())
    win_count = 0
    for i in range(N_EPOCH):
        # Play a new game
        while not game_simulator.terminal:
            # Act at random on the first few games
            if i < 100:
                action = np.random.randint(N_ACTIONS)
            # Act at random with a fixed probability
            elif np.random.uniform() >= 0.9:
                action = np.random.randint(N_ACTIONS)
            # Act following the policy on the other games
            else:
                action = np.argmax(dqn_agent.act(sess, np.array([x_t])))
            # Act on the game simulator
            res = game_simulator.input_key(action)
            # Receive the results from the game simulator
            x_t = res["state_prev"]
            x_t_plus_1 = res["state"]
            terminal = res["terminal"]
            r_t = res["reward"]
            a_t = res["action"]
            if i == 0 or r_t > 0.5:
                replay_memory.store(x_t, a_t, r_t, x_t_plus_1, terminal)
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
        print("epoch: {0} win_rate: {1} reward: {2} loss: {3}".format(
            i, win_count/(i+1e-5), r_t, np.mean(train_loss))
        )
        if r_t > 0.5:
            win_count += 1
        game_simulator.init_game()
    # Save model
    dqn_agent.save_model(sess, "savetest")
