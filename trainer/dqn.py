# -*- coding: utf-8 -*-

# Default modules
import logging

# Additional modules
import numpy as np
import tensorflow as tf

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class DQN:

    def __init__(self, input_size=84, learning_rate=1e-4, n_actions=5):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.gamma = 0.95
        self.x_ph = tf.placeholder(tf.float32, shape=[None, input_size], name="x_placeholder")
        self.y_ph = tf.placeholder(tf.float32, shape=[None], name="y_placeholder")
        self.a_ph = tf.placeholder(tf.int64, shape=[None], name="a_placeholder")
        self.q = self._inference(self.x_ph, self.n_actions)
        self.loss = self._build_loss(self.y_ph, self.q, self.a_ph)
        self.train_ops = self._build_optimizer(self.loss, learning_rate)
        self.merged = tf.merge_all_summaries()

    @staticmethod
    def _inference(x_ph, n_actions):
        logits = tf.contrib.layers.fully_connected(x_ph, n_actions, activation_fn=None)
        outputs = tf.nn.softmax(logits)
        return outputs

    @staticmethod
    def _build_loss(y_t_ph, q_t, a_ph):
        with tf.name_scope("loss"):
            a_t_one_hot = tf.one_hot(a_ph, q_t.get_shape()[1].value)
            q_t_acted = tf.reduce_sum(q_t * a_t_one_hot, reduction_indices=1)
            loss = tf.squared_difference(q_t_acted, y_t_ph)
        return loss

    @staticmethod
    def _build_optimizer(loss, learning_rate):
        train_ops = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
        return train_ops

    def update(self, sess, x_t, a_t, r_t, x_t_plus_1, terminal):
        # Compute target score
        fd = {self.x_ph: x_t_plus_1}
        q_t_plus_1 = np.argmax(sess.run(self.q, feed_dict=fd), axis=1)
        y_t = r_t + q_t_plus_1 * (1-terminal) * self.gamma
        # Run optimization operation
        fd = {self.x_ph: x_t, self.y_ph: y_t, self.a_ph: a_t}
        _, train_loss = sess.run([self.train_ops, self.loss], feed_dict=fd)
        return train_loss

    def act(self, sess, x_t):
        return sess.run(self.q, feed_dict={self.x_ph: x_t})

    def write_summary(self, sess):
        return sess.run(self.merged)

    # TODO: model saver
    def save_model(self):
        return 0
