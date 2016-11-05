# -*- coding: utf-8 -*-

import numpy as np


class ChasingSimulator:

    def __init__(self, field_size):
        self.field_size = field_size
        self.iter = 0
        self.player_position = None
        self.enemy_position = None
        self.init_game()

    def init_game(self):
        self.iter = 0
        # Decide player position at random
        self.player_position = np.array([
            np.random.randint(self.field_size),
            np.random.randint(self.field_size)
        ])
        # Decide enemy position at random
        self.enemy_position = np.array([
            np.random.randint(self.field_size),
            np.random.randint(self.field_size)
        ])
        # TODO: Generate walls at random here

    def input_key(self, action):
        state_prev = self.draw_field()
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
            terminal = True
            reward = 1
        # Time over
        elif self.iter >= 2 * self.field_size:
            terminal = True
            reward = -1
        # Give reward for distance
        else:
            terminal = False
            reward = -np.sum(np.abs(self.player_position-self.enemy_position)) / (2.*self.field_size)
        result = {
            "reward": reward,
            "state_prev": state_prev,
            "state": self.draw_field(),
            "terminal": terminal,
            "action": action
        }
        self.iter += 1
        if terminal:
            self.init_game()
        return result

    def draw_field(self):
        field = np.zeros((self.field_size, self.field_size), dtype=np.int32)
        field[self.player_position[0], self.player_position[1]] = 1
        field[self.enemy_position[0], self.enemy_position[1]] = -1
        return field
