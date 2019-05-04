import gym
import tensorflow as tf
from . import agents


def q_fn(x, n_actions):
    hidden = tf.layers.dense(x, 64, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden, n_actions, activation=None)
    return outputs


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    env = gym.make("cartpole-v1")
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=1e-2)
    agent = agents.DQN(
        q_fn=q_fn,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        optimizer=optimizer,
    )
    agents.train_and_play_game(
        agent=agent,
        env=env,
        max_episodes=1000,
        max_frames=1000000,
        replay_memory_size=200000,
        update_frequency=1,
        target_sync_frequency=1000,
        final_exploration_frame=10000,
        action_repeat=1,
        batch_size=32,
        # checkpoint_dir="outputs/cartpole-v0",
    )


if __name__ == "__main__":
    main()
