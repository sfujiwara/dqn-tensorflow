# -*- coding: utf-8 -*-

import numpy as np
from gym import Env, spaces


class MetalTileEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, field_size=84):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-1, 1, [field_size, field_size, 3])
        self.field_size = field_size
        self.iter = 0
        self.terminal = False
        self.player_position = None
        self.enemy_position = None
        self.reset()

    def _reset(self):
        self.terminal = False
        self.iter = 0
        # Decide player position at random
        self.player_position = np.random.randint(low=0, high=self.field_size, size=2)
        # Decide enemy position at random
        self.enemy_position = np.random.randint(low=0, high=self.field_size, size=2)
        # TODO: Generate walls at random here
        return self.state()

    def _step(self, action):
        # Update player position
        # Do nothing
        if action == 0:
            pass
        # Left
        elif action == 1:
            self.player_position[1] = max(0, self.player_position[1] - 1)
        # Right
        elif action == 2:
            self.player_position[1] = min(self.field_size - 1, self.player_position[1] + 1)
        # Up
        elif action == 3:
            self.player_position[0] = max(0, self.player_position[0]-1)
        # Down
        elif action == 4:
            self.player_position[0] = min(self.field_size - 1, self.player_position[0] + 1)
        else:
            raise ValueError
        # TODO: update enemy position
        # Game clear
        if self.player_position[0] == self.enemy_position[0] and self.player_position[1] == self.enemy_position[1]:
            self.terminal = True
            reward = 1.
        # Time over
        elif self.iter >= 5 * self.field_size:
            self.terminal = True
            reward = self.compute_reward()
            # reward = 0.
        # Give reward for distance
        else:
            self.terminal = False
            reward = self.compute_reward()
            # reward = 0
        self.iter += 1
        info = None
        return self.state(), reward, self.terminal, info

    def _render(self, mode="human", close=False):
        field = np.array([["."]*self.field_size]*self.field_size)
        field[self.player_position[0], self.player_position[1]] = "P"
        field[self.enemy_position[0], self.enemy_position[1]] = "E"
        field_str = ""
        for i in field:
            for j in i:
                field_str += j
                field_str += "\t"
            field_str += "\n"
        print(field_str)

    def compute_reward(self):
        # dist = np.linalg.norm(self.player_position - self.enemy_position)
        # max_dist = np.linalg.norm([self.field_size, self.field_size])
        # reward = - dist / max_dist / 0.1 / self.field_size
        x_diff = np.abs(self.player_position[0] - self.enemy_position[0])
        y_diff = np.abs(self.player_position[1] - self.enemy_position[1])
        reward = -(x_diff + y_diff) / (self.field_size * 2. * 1000.)
        return reward

    def state(self):
        s = np.zeros([self.field_size, self.field_size, 3], dtype=np.float32)
        # Set player position on first kernel
        s[self.player_position[0], self.player_position[1], 0] = 1
        # Set enemy position on second kernel
        s[self.enemy_position[0], self.enemy_position[1], 1] = 1
        # TODO: Set structures on third kernel
        return s
        # return s.flatten()

if __name__ == "__main__":
    env = MetalTileEnv(field_size=4)
    n_action = 5
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = np.random.randint(n_action)
        observation, reward, done, info = env.step(action)
        print(reward)
