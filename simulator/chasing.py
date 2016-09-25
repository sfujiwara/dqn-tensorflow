# -*- coding: utf-8 -*-

import numpy as np


class ChasingSimulator:

    def __init__(self, field_size=84):
        self.field_size = field_size
        self.iter = 0
        self.player_position = None
        self.enemy_position = None
        self.init_field()

    def init_field(self):
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
        if action == "right":
            self.player_position[1] = min(self.field_size - 1, self.player_position[1] + 1)
        elif action == "left":
            self.player_position[1] = max(0, self.player_position[1]-1)
        elif action == "up":
            self.player_position[0] = max(0, self.player_position[0]-1)
        elif action == "down":
            self.player_position[0] = min(self.field_size - 1, self.player_position[0] + 1)
        elif action == "do_nothing":
            pass
        else:
            raise ValueError
        # TODO: update enemy position
        # Compute reward
        # print self.player_position, self.enemy_position, action
        if np.all(np.abs(self.player_position-self.enemy_position) <= 1):
            terminal = True
            reward = 1
        elif self.iter >= 50:
            terminal = True
            reward = -1
        else:
            terminal = False
            norm = np.linalg.norm(self.player_position - self.enemy_position)
            reward = -norm/10000.
        result = {
            "reward": reward,
            "state_prev": state_prev,
            "state": self.draw_field(),
            "terminal": terminal,
            "action": action
        }
        self.iter += 1
        if terminal:
            self.init_field()
        return result

    def draw_field(self):
        field = np.zeros((self.field_size, self.field_size), dtype=np.int32)
        field[self.player_position[0], self.player_position[1]] = 1
        field[self.enemy_position[0], self.enemy_position[1]] = -1
        return field
