import numpy as np
import tensorflow as tf

from . import repmem


class DQN:

    def __init__(
            self,
            input_shape,
            n_actions,
            q_fn,
            learning_rate,
            discount_factor=0.99
    ):
        """
        Parameters
        ----------
        input_shape: the shape of input stat
            - type: list of int
            - example: [84, 84, 4] for Atari game in the original DQN paper

        n_actions: the number of actions the agent can choose
            - type: int

        q_fn: a function building the computation graph for q-network
            - type: callable
            - input of q_fn: Tensor of shape [None, input_shape[0], input_shape[1], ...] and n_actions
            - output of q_fn: Tensor of shape [None, n_actions]

        learning_rate: the step size of the optimization method
            - type: float
        """
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.gamma = discount_factor
        self.input_shape = input_shape
        self.q_fn = q_fn
        # References to graph nodes are assigned after running `build_graph` method
        self.x_ph, self.y_ph, self.a_ph = None, None, None
        self.q, self.loss, self.train_ops = None, None, None
        self.target_x_ph, self.target_q = None, None
        self.assign_ops = None

    def build_graph(self):
        # Create placeholders
        self.x_ph = tf.placeholder(tf.float32, shape=[None]+list(self.input_shape), name="x_ph")
        self.y_ph = tf.placeholder(tf.float32, shape=[None], name="y_ph")
        self.a_ph = tf.placeholder(tf.int64, shape=[None], name="a_ph")
        # Build q network
        with tf.variable_scope("qnet"):
            self.q = self.q_fn(self.x_ph, self.n_actions)
        self.loss = self._build_loss(self.y_ph, self.q, self.a_ph)
        # Build target q network
        self.target_x_ph = tf.placeholder(tf.float32, shape=[None] + list(self.input_shape), name="target_x_ph")
        with tf.variable_scope("target_qnet"):
            self.target_q = self.q_fn(self.target_x_ph, self.n_actions)
        # Build update target q-network ops
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="qnet")
        target_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_qnet")
        self.train_ops = self._build_optimizer(self.loss, self.learning_rate)
        self.assign_ops = [tf.assign(target_q_vars[i], q_vars[i]) for i in range(len(q_vars))]

    @staticmethod
    def _build_loss(y_t_ph, q_t, a_ph):
        with tf.name_scope("loss"):
            a_t_one_hot = tf.one_hot(a_ph, q_t.get_shape()[1].value)
            q_t_acted = tf.reduce_sum(q_t * a_t_one_hot, reduction_indices=1)
            loss = tf.losses.mean_squared_error(labels=y_t_ph, predictions=q_t_acted)
        return loss

    @staticmethod
    def _build_optimizer(loss, learning_rate):
        global_step = tf.train.get_or_create_global_step()
        optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.95, epsilon=1e-2)
        # optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train_op = optim.minimize(loss, global_step=global_step)
        return train_op

    def update(self, sess, x_t, a_t, r_t, x_t_plus_1, terminal):
        # Compute target score
        fd = {self.target_x_ph: x_t_plus_1}
        q_t_plus_1 = np.max(sess.run(self.target_q, feed_dict=fd), axis=1)
        y_t = r_t + q_t_plus_1 * (1-terminal) * self.gamma
        # Run optimization operation
        fd = {self.x_ph: x_t, self.y_ph: y_t, self.a_ph: a_t}
        _, train_loss = sess.run([self.train_ops, self.loss], feed_dict=fd)
        return train_loss

    def act(self, sess, x_t):
        return sess.run(self.q, feed_dict={self.x_ph: x_t})

    def update_target_q_network(self, sess):
        sess.run(self.assign_ops)


def train_and_play_game(
        agent,
        env,
        max_episodes,
        replay_memory_size,
        batch_size,
        update_frequency,
        target_sync_frequency,
        final_exploration_frame,
        log_frequency=20,
        action_repeat=4,
        max_no_op=30,
        checkpoint_dir=None,
):
    """
    Parameters
    ----------

    """
    replay_memory = repmem.ReplayMemory(memory_size=replay_memory_size)
    total_reward_list = []
    with tf.Graph().as_default() as g:
        agent.build_graph()
        episode_count = step_count = action_count = frame_count = 0

        with tf.train.MonitoredTrainingSession(
            save_summaries_steps=100,
            checkpoint_dir=checkpoint_dir,
        ) as mon_sess:

            # Training loop
            while episode_count < max_episodes:
                random_action_prob = max(1 - float(frame_count)/final_exploration_frame*0.95, 0.05)
                # Play a new game
                previous_observation = env.reset()
                done = False
                total_reward = 0
                # Initial action
                action = np.random.randint(agent.n_actions)

                while not done:
                    # Act at random in first some frames
                    # for _ in range(np.random.randint(1, max_no_op)):
                    #     previous_observation, _, _, _ = env.step(env.action_space.sample())
                    # print(episode_count, step_count, action_count, frame_count)

                    # Update target Q network
                    if frame_count % target_sync_frequency == 0:
                        agent.update_target_q_network(mon_sess)

                    # Frame skip
                    if frame_count % action_repeat == 0:

                        # Act at random with a fixed probability
                        if np.random.rand() <= random_action_prob:
                            action = np.random.randint(agent.n_actions)
                        # Act following the policy on the other games
                        else:
                            q = agent.act(mon_sess, np.array([previous_observation]))
                            action = q.argmax()

                        action_count += 1

                    # Receive the results from the game simulator
                    observation, reward, done, info = env.step(action)
                    total_reward += reward

                    # Store the experience
                    if frame_count % action_repeat == 0:
                        replay_memory.store(previous_observation, action, reward, observation, done)

                    previous_observation = observation
                    # Update q network every update_interval
                    if action_count % update_frequency == 0:
                        mini_batch = replay_memory.sample(size=batch_size)
                        train_loss = agent.update(
                            mon_sess, mini_batch[0], mini_batch[1], mini_batch[2], mini_batch[3], mini_batch[4]
                        )
                        step_count += 1
                    frame_count += 1
                episode_count += 1
                total_reward_list.append(total_reward)

                # Show log every log_interval
                if episode_count % log_frequency == 0:
                    tf.logging.info("\t".join([
                        "Frame: {}".format(frame_count),
                        "Episode: {}".format(episode_count),
                        "Average Reward: {:.4f}".format(np.mean(total_reward_list[-50:])),
                        "Train Loss: {:.4f}".format(np.mean(train_loss)),
                        "Epsilon: {:.4f}".format(random_action_prob),
                    ]))
