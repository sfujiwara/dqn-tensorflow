# -*- coding: utf-8 -*-

import numpy as np


class SimpleChasingSimulator:

    def __init__(self, field_size=5):
        self.field_size = field_size
        self.iter = 0
        self.player_position = None
        self.enemy_position = None
        self.init_game()
        self.terminal = False

    def init_game(self):
        self.terminal = False
        self.iter = 0
        # Decide player position at random
        self.player_position = 2
        # Decide enemy position at random
        if np.random.uniform() > 0.5:
            self.enemy_position = 0
        else:
            self.enemy_position = 4

    def input_key(self, action):
        state_prev = self.draw_field()
        # Move left
        if action == 0:
            self.player_position = max(0, self.player_position-1)
        # Move right
        elif action == 1:
            self.player_position = min(self.field_size-1, self.player_position+1)
        # Game clear
        if self.player_position == self.enemy_position:
            self.terminal = True
            reward = 1.
        # Time over
        elif self.iter >= 2 * self.field_size:
            self.terminal = True
            reward = -0.1
        # Give reward for distance
        else:
            self.terminal = False
            reward = -0.
        result = {
            "reward": reward,
            "state_prev": state_prev,
            "state": self.draw_field(),
            "terminal": self.terminal,
            "action": action
        }
        self.iter += 1
        # if self.terminal:
        #     self.init_game()
        return result

    def draw_field(self):
        res = np.zeros(self.field_size)
        res[self.player_position] = 1
        res[self.enemy_position] = -1
        return res