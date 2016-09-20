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

    def __init__(self):
        self.gamma = 0.95
        # Placeholder: [batch_size x height x width x n_channels]
        self.x_ph = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name="x_placeholder")
        self.y_ph = tf.placeholder(tf.float32, shape=[None], name="y_placeholder")
        logger.info("build computation graph")
        self.q = self._inference(self.x_ph)
        self.loss = self._build_loss(self.y_ph, self.q)
        self.train_ops = self._build_optimizer(self.loss)

    @staticmethod
    def _inference(x_ph):
        with tf.name_scope("Q_function"):
            w_conv1 = tf.Variable(tf.truncated_normal(shape=[8, 8, 4, 16], stddev=0.001))
            b_conv1 = tf.Variable(tf.constant(0., shape=[16]))
            h_conv1 = tf.nn.relu6(
                tf.nn.bias_add(
                    tf.nn.conv2d(x_ph, w_conv1, strides=[1, 4, 4, 1], padding="SAME"),
                    b_conv1
                )
            )
            w_conv2 = tf.Variable(tf.truncated_normal(shape=[4, 4, 16, 32], stddev=0.001))
            b_conv2 = tf.Variable(tf.constant(0., shape=[32]))
            h_conv2 = tf.nn.relu6(
                tf.nn.bias_add(
                    tf.nn.conv2d(h_conv1, w_conv2, strides=[1, 2, 2, 1], padding="SAME"),
                    b_conv2
                )
            )
            h_conv2_flat_t = tf.reshape(h_conv2, shape=[-1, 11*11*32])
            w_fc1 = tf.Variable(tf.truncated_normal(shape=[11*11*32, 256], stddev=0.001))
            b_fc1 = tf.Variable(tf.constant(0., shape=[256]))
            h_fc1 = tf.nn.bias_add(tf.matmul(h_conv2_flat_t, w_fc1), b_fc1)
            w_fc2 = tf.Variable(tf.truncated_normal(shape=[256, 4], stddev=0.001))
            b_fc2 = tf.Variable(tf.constant(0., shape=[4]))
            outputs = tf.nn.bias_add(tf.matmul(h_fc1, w_fc2), b_fc2)
        return outputs

    @staticmethod
    def _build_loss(y_t_ph, q_t):
        total_reward_t = tf.reduce_max(q_t, reduction_indices=1)
        loss = tf.squared_difference(total_reward_t, y_t_ph)
        return loss

    @staticmethod
    def _build_optimizer(loss):
        train_ops = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss)
        return train_ops

    def update(self, sess, x_t, x_t_plus_1, r_t, terminal):
        y_t = r_t + np.max(sess.run(self.q, feed_dict={self.x_ph: x_t_plus_1}), axis=1) * (1-terminal)
        fd = {
            self.x_ph: x_t,
            self.y_ph: y_t,
        }
        _, train_loss = sess.run([self.train_ops, self.loss], feed_dict=fd)
        return train_loss

    def act(self, sess, x_t):
        return sess.run(self.q, feed_dict={self.x_ph: x_t})
