import gym
import tensorflow as tf
from . import breakout_env
from . import agents


def q_fn(x, n_actions):
    with tf.variable_scope("hidden1"):
        hidden1 = tf.layers.dense(x, 128, activation=tf.nn.relu)
    with tf.variable_scope("hidden2"):
        hidden2 = tf.layers.dense(hidden1, 128, activation=tf.nn.relu)
    with tf.variable_scope("output"):
        outputs = tf.layers.dense(hidden2, n_actions, activation=None)
    return outputs


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    env = gym.make("CartPole-v1")
    agent = agents.DQN(
        q_fn=q_fn,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        learning_rate=0.001
    )
    agents.train_and_play_game(
        agent=agent,
        env=env,
        random_action_decay=0.999,
        max_episodes=3000,
        replay_memory_size=1000,
        batch_size=32,
        n_updates_on_episode=20,
    )


if __name__ == "__main__":
    main()
