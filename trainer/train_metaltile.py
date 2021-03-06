import argparse
import tensorflow as tf
from . import metaltile
from . import agents


parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)
parser.add_argument("--learning_rate", type=float, default=0.00025)
parser.add_argument("--replay_memory_size", type=int, default=200000)
parser.add_argument("--update_frequency", type=int, default=4)
parser.add_argument("--checkpoint_dir", type=str, default=None)

args, unknown_args = parser.parse_known_args()

REPLAY_MEMORY_SIZE = args.replay_memory_size
LEARNING_RATE = args.learning_rate
UPDATE_FREQUENCY = args.update_frequency
CHECKPOINT_DIR = args.checkpoint_dir


def q_fn(x, n_actions):
    x = tf.layers.flatten(x)
    x = tf.layers.dense(
        x, 128, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01)
    )
    x = tf.layers.dense(
        x, 128, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01)
    )
    outputs = tf.layers.dense(x, n_actions, activation=None)
    return outputs


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    env = metaltile.MetalTileEnv(field_size=8, time_limit=80)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, momentum=0.95, epsilon=1e-2)
    optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.9, use_nesterov=True)
    agent = agents.DQN(
        q_fn=q_fn,
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        optimizer=optimizer,
    )
    agents.train_and_play_game(
        agent=agent,
        env=env,
        max_episodes=150000,
        max_frames=1000000,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        batch_size=32,
        update_frequency=UPDATE_FREQUENCY,
        target_sync_frequency=1000,
        final_exploration_frame=500000,
        action_repeat=1,
        max_no_op=0,
        checkpoint_dir=CHECKPOINT_DIR,
    )


if __name__ == "__main__":
    main()
