# -*- coding: utf-8 -*-

import numpy as np


class ChasingSimulator:

    def __init__(self, field_size=84):
        self.field_size = field_size
        self.iter = 0
        self.terminal = False
        self.player_position = None
        self.enemy_position = None
        self.reset()

    def reset(self):
        """
        Initialize player position, enemy position, structure, and iteration.
        :return: None
        """
        self.terminal = False
        self.iter = 0
        # Decide player position at random
        self.player_position = np.random.randint(low=0, high=self.field_size, size=2)
        # Decide enemy position at random
        self.enemy_position = np.random.randint(low=0, high=self.field_size, size=2)
        # TODO: Generate walls at random here
        return self.state()

    def step(self, action):
        state_prev = self.state()
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
        if np.all(self.player_position == self.enemy_position):
            self.terminal = True
            reward = 1.
        # Time over
        elif self.iter >= 5 * self.field_size:
            self.terminal = True
            reward = self.compute_reward()
        # Give reward for distance
        else:
            self.terminal = False
            reward = self.compute_reward()
        self.iter += 1
        info = None
        return self.state(), reward, self.terminal, info

    def compute_reward(self):
        dist = np.linalg.norm(self.player_position - self.enemy_position)
        max_dist = np.linalg.norm([self.field_size, self.field_size])
        reward = - dist / max_dist * 10
        return reward

    def render(self):
        field = np.array([['-']*self.field_size]*self.field_size)
        field[self.player_position[0], self.player_position[1]] = 'P'
        field[self.enemy_position[0], self.enemy_position[1]] = 'E'
        print(field)

    def state(self):
        s = np.zeros([self.field_size, self.field_size, 3], dtype=np.float32)
        # Set player position on first kernel
        s[self.player_position[0], self.player_position[1], 0] = 1
        # Set enemy position on second kernel
        s[self.enemy_position[0], self.enemy_position[1], 1] = 1
        # TODO: Set structures on third kernel
        return s

if __name__ == "__main__":
    env = ChasingSimulator(field_size=4)
    n_action = 4
    # Use env same as OpenAI Gym: https://gym.openai.com
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = np.random.randint(n_action)
        observation, reward, done, info = env.step(action)
