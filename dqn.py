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

    def __init__(
            self,
            input_size=84,
            n_channels=4,
            learning_rate=1e-2,
            stride=[1, 1],
    ):
        self.n_channels = n_channels
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.stride = stride
        self.gamma = 0.95
        # Placeholder: [batch_size x height x width x n_channels]
        self.x_ph = tf.placeholder(
            tf.float32,
            shape=[None, self.input_size, self.input_size, self.n_channels],
            name="x_placeholder"
        )
        self.y_ph = tf.placeholder(tf.float32, shape=[None], name="y_placeholder")
        logger.info("build computation graph")
        self.q = self._inference(self.x_ph)
        self.loss = self._build_loss(self.y_ph, self.q)
        self.train_ops = self._build_optimizer(self.loss)

    def _inference(self, x_ph):
        with tf.name_scope("Q_function"):
            w_conv1 = tf.Variable(tf.truncated_normal(
                shape=[8, 8, self.n_channels, 16], stddev=0.01)
            )
            b_conv1 = tf.Variable(tf.constant(0., shape=[16]))
            h_conv1 = tf.nn.relu6(
                tf.nn.bias_add(
                    tf.nn.conv2d(x_ph, w_conv1, strides=[1, self.stride[0], self.stride[0], 1], padding="SAME"),
                    b_conv1
                )
            )
            w_conv2 = tf.Variable(tf.truncated_normal(shape=[4, 4, 16, 32], stddev=0.01))
            b_conv2 = tf.Variable(tf.constant(0., shape=[32]))
            h_conv2 = tf.nn.relu6(
                tf.nn.bias_add(
                    tf.nn.conv2d(h_conv1, w_conv2, strides=[1, self.stride[1], self.stride[1], 1], padding="SAME"),
                    b_conv2
                )
            )
            n_flat = int(np.ceil(np.ceil(float(self.input_size) / self.stride[0]) / self.stride[1]))**2 * 32
            h_conv2_flat_t = tf.reshape(h_conv2, shape=[-1, n_flat])
            w_fc1 = tf.Variable(tf.truncated_normal(shape=[n_flat, 256], stddev=0.01))
            b_fc1 = tf.Variable(tf.constant(0., shape=[256]))
            h_fc1 = tf.nn.bias_add(tf.matmul(h_conv2_flat_t, w_fc1), b_fc1)
            w_fc2 = tf.Variable(tf.truncated_normal(shape=[256, 5], stddev=0.01))
            b_fc2 = tf.Variable(tf.constant(0., shape=[5]))
            outputs = tf.nn.bias_add(tf.matmul(h_fc1, w_fc2), b_fc2)
        return outputs

    @staticmethod
    def _build_loss(y_t_ph, q_t):
        total_reward_t = tf.reduce_max(q_t, reduction_indices=1)
        loss = tf.squared_difference(total_reward_t, y_t_ph)
        return loss

    def _build_optimizer(self, loss):
        train_ops = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return train_ops

    def update(self, sess, x_t, x_t_plus_1, r_t, terminal):
        # Compute target score
        fd = {self.x_ph: x_t_plus_1}
        y_t = r_t + np.max(sess.run(self.q, feed_dict=fd), axis=1) * (1-terminal) * self.gamma
        # Run optimization operation
        fd = {self.x_ph: x_t, self.y_ph: y_t}
        _, train_loss = sess.run([self.train_ops, self.loss], feed_dict=fd)
        return train_loss

    def act(self, sess, x_t):
        return sess.run(self.q, feed_dict={self.x_ph: x_t})
