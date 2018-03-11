import collections
import gym
from gym import Env, spaces
import numpy as np
from scipy.misc import imresize
from skimage.color import rgb2gray


class BreakoutEnv(Env):
    """
    Note:
        - If agent act at random, it takes about 200~300 frames per one episode
    """

    def __init__(self):
        self.breakout_v0_env = gym.make("Breakout-v0")
        self.action_space = self.breakout_v0_env.action_space
        self.observation_space = spaces.Box(low=0, high=1, shape=(84, 84, 4))
        self.recent_observations = collections.deque(maxlen=4)

    def reset(self):
        observation = self.breakout_v0_env.reset()
        observation = self._to_84x84_grayscale(observation)
        for _ in range(4):
            self.recent_observations.append(observation)
        return np.stack(self.recent_observations, 2)

    def step(self, action):
        observation, reward, done, info = self.breakout_v0_env.step(action)
        self.recent_observations.append(self._to_84x84_grayscale(observation))
        return np.stack(self.recent_observations, 2), reward, done, info

    @staticmethod
    def _to_84x84_grayscale(observation):
        resized_observation = imresize(observation, [110, 84], interp="nearest")[17:101]
        resized_grayscale_observation = rgb2gray(resized_observation)
        return resized_grayscale_observation


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = BreakoutEnv()
    observation = env.reset()
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break

    img = np.hstack([observation[:, :, 0], observation[:, :, 1], observation[:, :, 2], observation[:, :, 3]])
    plt.imshow(img, cmap="gray")
    plt.show()
