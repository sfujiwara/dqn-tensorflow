import tensorflow as tf

from . import breakout_env
from . import agents


def q_fn(x, n_actions):
    conv1 = tf.layers.conv2d(
        inputs=x, filters=32, kernel_size=[8, 8],
        strides=4, activation=tf.nn.relu, name="conv1"
    )
    conv2 = tf.layers.conv2d(
        inputs=conv1, filters=64, kernel_size=[4, 4],
        strides=2, activation=tf.nn.relu, name="conv2"
    )
    conv3 = tf.layers.conv2d(
        inputs=conv2, filters=64, kernel_size=[3, 3],
        strides=1, activation=tf.nn.relu, name="conv3"
    )
    conv3_flat = tf.layers.flatten(conv3)
    fc = tf.layers.dense(inputs=conv3_flat, units=512, activation=tf.nn.relu, name="fc1")
    outputs = tf.layers.dense(inputs=fc, units=n_actions, name="output")
    return outputs


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    env = breakout_env.BreakoutEnv()
    agent = agents.DQN(
        q_fn=q_fn,
        input_shape=[84, 84, 4],
        n_actions=env.action_space.n,
        learning_rate=0.00025
    )
    agents.train_and_play_game(
        agent=agent,
        env=env,
        random_action_decay=0.999,
        max_episodes=15000,
        replay_memory_size=int(1000000/5),
        batch_size=32,
        update_frequency=4,
        target_sync_frequency=10000,
        final_exploration_frame=1000000,
        action_repeat=1,
        max_no_op=0,
        checkpoint_dir="outputs/breakout-v0"
    )


if __name__ == "__main__":
    main()
