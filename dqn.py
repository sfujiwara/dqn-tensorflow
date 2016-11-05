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

    def __init__(self, input_size=84, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.gamma = 0.95
        self.x_ph = tf.placeholder(tf.float32, shape=[None, input_size**2], name="x_placeholder")
        self.y_ph = tf.placeholder(tf.float32, shape=[None], name="y_placeholder")
        self.a_ph = tf.placeholder(tf.int64, shape=[None], name="a_placeholder")
        self.q = self._inference(self.x_ph)
        self.loss = self._build_loss(self.y_ph, self.q, self.a_ph)
        self.train_ops = self._build_optimizer(self.loss, learning_rate)
        self.merged = tf.merge_all_summaries()

    @staticmethod
    def _inference(x_ph):
        with tf.name_scope("q_network/hidden1"):
            # Hidden layer 1
            w_fc1 = tf.Variable(tf.truncated_normal([x_ph.get_shape()[1].value, 8]))
            b_fc1 = tf.Variable(tf.zeros([8]))
            h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x_ph, w_fc1), b_fc1))
            # Summaries
            tf.histogram_summary(w_fc1.name, w_fc1)
            tf.histogram_summary(b_fc1.name, b_fc1)
        with tf.name_scope("q_network/output"):
            # Hidden layer 2
            w_out = tf.Variable(tf.truncated_normal([8, 5]))
            b_out = tf.Variable(tf.zeros([5]))
            outputs = tf.nn.bias_add(tf.matmul(h_fc1, w_out), b_out)
            # Summaries
            tf.histogram_summary(w_out.name, w_out)
            tf.histogram_summary(b_out.name, b_out)
        return outputs

    @staticmethod
    def _build_loss(y_t_ph, q_t, a_ph):
        with tf.name_scope("loss"):
            a_t_one_hot = tf.one_hot(a_ph, 5)
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
