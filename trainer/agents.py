# TODO: implement target q network

# Default modules
import os

# Additional modules
import numpy as np
import tensorflow as tf

from . import repmem


class DQN:

    def __init__(
            self,
            input_shape,
            n_actions,
            q_fn,
            learning_rate=1e-2,
    ):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.gamma = 0.95
        self.input_shape = input_shape
        self.q_fn = q_fn
        # Reference to graph is assigned after running `build_graph` method
        self.x_ph, self.y_ph, self.a_ph = None, None, None
        self.q, self.loss, self.train_ops = None, None, None

    def build_graph(self):
        # Create placeholders
        self.x_ph = tf.placeholder(tf.float32, shape=[None]+list(self.input_shape), name="x_ph")
        self.y_ph = tf.placeholder(tf.float32, shape=[None], name="y_ph")
        self.a_ph = tf.placeholder(tf.int64, shape=[None], name="a_ph")
        # Build q network
        self.q = self.q_fn(self.x_ph, self.n_actions)
        self.loss = self._build_loss(self.y_ph, self.q, self.a_ph)
        self.train_ops = self._build_optimizer(self.loss, self.learning_rate)

    @staticmethod
    def _build_loss(y_t_ph, q_t, a_ph):
        with tf.name_scope("loss"):
            a_t_one_hot = tf.one_hot(a_ph, q_t.get_shape()[1].value)
            q_t_acted = tf.reduce_sum(q_t * a_t_one_hot, reduction_indices=1)
            loss = tf.losses.mean_squared_error(labels=y_t_ph, predictions=q_t_acted)
        return loss

    @staticmethod
    def _build_optimizer(loss, learning_rate):
        train_op = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            decay=0.9,
            momentum=0.95
        ).minimize(loss)
        # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return train_op

    def update(self, sess, x_t, a_t, r_t, x_t_plus_1, terminal):
        # Compute target score
        fd = {self.x_ph: x_t_plus_1}
        q_t_plus_1 = np.max(sess.run(self.q, feed_dict=fd), axis=1)
        # print q_t_plus_1
        y_t = r_t + q_t_plus_1 * (1-terminal) * self.gamma
        # Run optimization operation
        fd = {self.x_ph: x_t, self.y_ph: y_t, self.a_ph: a_t}
        _, train_loss = sess.run([self.train_ops, self.loss], feed_dict=fd)
        return train_loss

    def act(self, sess, x_t):
        return sess.run(self.q, feed_dict={self.x_ph: x_t})

    def save_model(self, dir):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(dir, "checkpoints"))
        model_dir = os.path.join(dir, "models", "episode-{}".format(latest_checkpoint.split("-")[-1]))
        # Save model for deployment on ML Engine
        with tf.Graph().as_default():
            input_key = tf.placeholder(tf.int64, [None, ], name="key")
            output_key = tf.identity(input_key)
            x_ph = tf.placeholder(tf.float32, shape=self.x_ph.get_shape(), name="x_ph")
            q = self._inference(x_ph, self.n_actions)
            saver = tf.train.Saver()
            input_signatures = {
                "key": tf.saved_model.utils.build_tensor_info(input_key),
                "state": tf.saved_model.utils.build_tensor_info(x_ph)
            }
            output_signatures = {
                "key": tf.saved_model.utils.build_tensor_info(output_key),
                "q": tf.saved_model.utils.build_tensor_info(q)
            }
            predict_signature_def = tf.saved_model.signature_def_utils.build_signature_def(
                input_signatures,
                output_signatures,
                tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
            builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
            with tf.Session() as sess:
                # Restore variables from latest checkpoint
                saver.restore(sess, latest_checkpoint)
                builder.add_meta_graph_and_variables(
                    sess=sess,
                    tags=[tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def
                    },
                    assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
                )
                builder.save()


def train_and_play_game(
        agent,
        env,
        random_action_decay,
        max_episodes,
        replay_memory_size,
        batch_size,
        n_updates_on_episode,
        update_interval=1,
):
    replay_memory = repmem.ReplayMemory(memory_size=replay_memory_size)
    total_reward_list = []
    with tf.Graph().as_default() as g:
        agent.build_graph()
        global_step = tf.Variable(0, trainable=False, name="global_step")
        increment_global_step_op = global_step.assign_add(1)
        with tf.train.MonitoredTrainingSession() as mon_sess:
            # Training loop
            while mon_sess.run(global_step) < max_episodes:
                random_action_prob = max(random_action_decay**mon_sess.run(global_step), 0.05)
                # Play a new game
                previous_observation = env.reset()
                done = False
                total_reward = 0
                while not done:
                    # Act at random with a fixed probability
                    if np.random.rand() <= random_action_prob:
                        action = np.random.randint(agent.n_actions)
                    # Act following the policy on the other games
                    else:
                        q = agent.act(mon_sess, np.array([previous_observation]))
                        action = q.argmax()
                    # Receive the results from the game simulator
                    observation, reward, done, info = env.step(action)
                    # reward = reward if not done else -10
                    total_reward += reward
                    # Store the experience
                    replay_memory.store(previous_observation, action, reward, observation, done)
                    previous_observation = observation
                total_reward_list.append(total_reward)

                # Update the policy
                for _ in range(n_updates_on_episode):
                    mini_batch = replay_memory.sample(size=batch_size)
                    train_loss = agent.update(
                        mon_sess, mini_batch[0], mini_batch[1], mini_batch[2], mini_batch[3], mini_batch[4]
                    )
                print(
                    "Episode: {0} Average Reward: {1} Training Loss: {2} Random Action Probability: {3}".format(
                        mon_sess.run(global_step), np.mean(total_reward_list[-50:]), np.mean(train_loss), random_action_prob)
                )
                mon_sess.run(increment_global_step_op)
