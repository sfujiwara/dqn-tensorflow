import tensorflow as tf

from . import metaltile
from . import agents


def q_fn(x, n_actions):
    x = tf.layers.flatten(x)
    hidden = tf.layers.dense(x, 64, activation=tf.nn.relu)
    outputs = tf.layers.dense(hidden, n_actions, activation=None)
    return outputs


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    env = metaltile.MetalTileEnv(field_size=8)
    print(env.observation_space.shape)
    agent = agents.DQN(
        q_fn=q_fn,
        input_shape=env.observation_space.shape,
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
        # checkpoint_dir="outputs/metaltile-v0"
    )


if __name__ == "__main__":
    main()
