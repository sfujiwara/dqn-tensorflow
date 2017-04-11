# -*- coding: utf-8 -*-

# Default modules
import json

# Additional modules
import numpy as np
import tensorflow as tf


class DQN:

    def __init__(self, input_size=84, learning_rate=1e-4, n_actions=5):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.gamma = 0.9
        # Build graph
        self.x_ph = tf.placeholder(tf.float32, shape=[None, input_size, input_size, 3], name="x_placeholder")
        self.y_ph = tf.placeholder(tf.float32, shape=[None], name="y_placeholder")
        self.a_ph = tf.placeholder(tf.int64, shape=[None], name="a_placeholder")
        self.q = self._inference(self.x_ph, self.n_actions)
        self.loss = self._build_loss(self.y_ph, self.q, self.a_ph)
        self.train_ops = self._build_optimizer(self.loss, learning_rate)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    @staticmethod
    def _inference(x_ph, n_actions, simple=True):
        if simple:
            x_flat = tf.contrib.layers.flatten(x_ph)
            hidden1 = tf.contrib.layers.fully_connected(x_flat, 32, activation_fn=tf.nn.relu)
            outputs = tf.contrib.layers.fully_connected(hidden1, n_actions, activation_fn=None)
            return outputs
        else:
            h_conv1 = tf.contrib.layers.convolution2d(inputs=x_ph, num_outputs=16, kernel_size=8, stride=4)
            h_conv2 = tf.contrib.layers.convolution2d(inputs=h_conv1, num_outputs=32, kernel_size=4, stride=2)
            h_conv2_flat = tf.contrib.layers.flatten(h_conv2)
            outputs = tf.contrib.layers.fully_connected(h_conv2_flat, n_actions, activation_fn=None)
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
        train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
        return train_op

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

    def save_model(self, session, dir):
        input_size = self.x_ph.get_shape()[1].value
        # Create a new graph for prediction
        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32, shape=[None, input_size, input_size, 3], name="x_placeholder")
            q = self._inference(x, self.n_actions)
            # Define key element
            input_key = tf.placeholder(tf.int64, [None, ], name="key")
            output_key = tf.identity(input_key)
            # Define API inputs/outpus object
            inputs = {"key": input_key.name, "state": x.name}
            outputs = {"key": output_key.name, "q": q.name}
            g.add_to_collection("inputs", json.dumps(inputs))
            g.add_to_collection("outputs", json.dumps(outputs))
            # Save model
            tf.train.Saver().export_meta_graph(filename="{}/model/export.meta".format(dir))
            self.saver.save(session, "{}/model/export".format(dir), write_meta_graph=False)
