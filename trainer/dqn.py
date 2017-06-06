# -*- coding: utf-8 -*-

# Default modules
import os

# Additional modules
import numpy as np
import tensorflow as tf


class DQN:

    def __init__(self, input_shape, n_actions, learning_rate=1e-2):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.gamma = 0.95
        # Build graph
        self.x_ph = tf.placeholder(tf.float32, shape=[None]+list(input_shape), name="x_ph")
        self.y_ph = tf.placeholder(tf.float32, shape=[None], name="y_ph")
        self.a_ph = tf.placeholder(tf.int64, shape=[None], name="a_ph")
        self.q = self._inference(self.x_ph, self.n_actions)
        self.loss = self._build_loss(self.y_ph, self.q, self.a_ph)
        self.train_ops = self._build_optimizer(self.loss, learning_rate)
        self.merged = tf.summary.merge_all()

    @staticmethod
    def _inference(x_ph, n_actions):
        # Use fully connected neural net
        if len(x_ph.get_shape()) == 2:
            with tf.variable_scope("hidden1"):
                hidden1 = tf.layers.dense(x_ph, 20, activation=tf.nn.relu)
                # hidden1 = tf.layers.dense(x_ph, 2048, activation=tf.nn.relu)
            with tf.variable_scope("hidden2"):
                hidden2 = tf.layers.dense(hidden1, 20, activation=tf.nn.relu)
                # hidden2 = tf.layers.dense(hidden1, 1024, activation=tf.nn.relu)
            with tf.variable_scope("output"):
                outputs = tf.layers.dense(hidden2, n_actions, activation=None)
            return outputs
        # Use convolutional neural net
        else:
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(inputs=x_ph, filters=16, kernel_size=[8, 8], strides=4, activation=tf.nn.relu)
            with tf.variable_scope("conv2"):
                conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
            with tf.name_scope("flatten"):
                conv2_flat = tf.reshape(conv2, [-1, np.prod(conv2.get_shape()[1:]).value])
            with tf.variable_scope("fc"):
                fc = tf.layers.dense(inputs=conv2_flat, units=256, activation=tf.nn.relu)
            with tf.variable_scope("output"):
                outputs = tf.layers.dense(inputs=fc, units=n_actions)
            return outputs

    @staticmethod
    def _build_loss(y_t_ph, q_t, a_ph):
        with tf.name_scope("loss"):
            a_t_one_hot = tf.one_hot(a_ph, q_t.get_shape()[1].value)
            q_t_acted = tf.reduce_sum(q_t * a_t_one_hot, reduction_indices=1)
            loss = tf.reduce_mean(tf.square(q_t_acted - y_t_ph))
            # loss = tf.squared_difference(q_t_acted, y_t_ph)
        return loss

    @staticmethod
    def _build_optimizer(loss, learning_rate):
        # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
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

    def write_summary(self, sess):
        return sess.run(self.merged)

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
