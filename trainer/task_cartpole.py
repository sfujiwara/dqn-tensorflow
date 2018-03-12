import gym
import tensorflow as tf
from . import breakout_env
from . import agents


def q_fn(x, n_actions):
    hidden = tf.layers.dense(x, 64, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden, n_actions, activation=None)
    return outputs


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    env = gym.make("CartPole-v1")
    agent = agents.DQN(
        q_fn=q_fn,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        learning_rate=0.025
    )
    agents.train_and_play_game(
        agent=agent,
        env=env,
        random_action_decay=0.99,
        max_episodes=1000,
        replay_memory_size=200000,
        update_frequency=1,
        action_repeat=1,
        batch_size=32,
    )


if __name__ == "__main__":
    main()
