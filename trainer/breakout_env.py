import collections
import gym
from gym import Env, spaces
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from skimage.color import rgb2gray


class BreakoutEnv(Env):

    def __init__(self):
        self.breakout_v0_env = gym.make("Breakout-v0")
        self.action_space = self.breakout_v0_env.action_space
        self.observation_space = None
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
    env = BreakoutEnv()
    observation = env.reset()
    for i in range(10):
        # ipython
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation.shape)
        if done:
            break

    # observation = imresize(observation, [110, 84], interp="nearest")[17:101]
    # observation = rgb2gray(observation)
    img = np.hstack([observation[:, :, 0], observation[:, :, 1], observation[:, :, 2], observation[:, :, 3]])
    plt.imshow(img, cmap="gray")
    plt.show()
