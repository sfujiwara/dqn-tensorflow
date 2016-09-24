# -*- coding: utf-8 -*-

import numpy as np


class ChasingSimulator:

    def __init__(self):
        self.field_size = 8
        self.player_position = np.array([
            self.field_size - 1,
            # np.random.randint(self.board_size),
            np.random.randint(self.field_size)
        ])
        self.ball_position = np.array([0, np.random.randint(self.field_size)])

    def input_key(self, key):
        # update player position
        if key == "left":
            self.player_position[1] = min(self.field_size - 1, self.player_position[1] + 1)
        elif key == "right":
            self.player_position[1] = max(0, self.player_position[1]-1)
        # update ball position
        self.ball_position[0] += 1
        # collision detection
        if self.ball_position[0] == self.field_size-1:
            if np.all(self.player_position == self.ball_position):
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        result = {
            "reward": reward,
            "state": self.draw_field(),
            "terminal": False
        }
        return result

    def draw_field(self):
        # reset screen
        board = np.zeros((self.field_size, self.field_size), dtype=np.int32)
        # draw player
        board[self.player_position[0], self.player_position[1]] = 1
        # draw ball
        board[self.ball_position[0], self.ball_position[1]] = 1
        return board


if __name__ == "__main__":
    cb = ChasingSimulator()
    print cb.input_key("left")
