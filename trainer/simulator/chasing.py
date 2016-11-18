# -*- coding: utf-8 -*-

import numpy as np


class ChasingSimulator:

    def __init__(self, field_size=84):
        self.field_size = field_size
        self.iter = 0
        self.terminal = False
        self.player_position = None
        self.enemy_position = None
        self.init_game()

    def init_game(self):
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

    def input_key(self, action):
        state_prev = self.state()
        # Update player position
        # Do nothing
        if action == 0:
            pass
        # Right
        elif action == 1:
            self.player_position[1] = min(self.field_size - 1, self.player_position[1] + 1)
        # Left
        elif action == 2:
            self.player_position[1] = max(0, self.player_position[1]-1)
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
            reward = 1
        # Time over
        elif self.iter >= 2 * self.field_size:
            self.terminal = True
            reward = -1
        # Give reward for distance
        else:
            self.terminal = False
            reward = -np.sum(np.abs(self.player_position-self.enemy_position)) / (2.*self.field_size)
        result = {
            "reward": reward,
            "state_prev": state_prev,
            "state": self.state(),
            "terminal": self.terminal,
            "action": action
        }
        self.iter += 1
        return result

    def draw_field(self):
        field = np.array([['-']*self.field_size]*self.field_size)
        field[self.player_position[0], self.player_position[1]] = 'P'
        field[self.enemy_position[0], self.enemy_position[1]] = 'E'
        return field

    def state(self):
        x = np.zeros([self.field_size, self.field_size, 2], dtype=np.int32)
        x[self.player_position[0], self.player_position[1], 0] = 1
        x[self.enemy_position[0], self.enemy_position[1], 1] = 1
        return x
