# -*- coding: utf-8 -*-

import numpy as np


class SampleEmulator:

    def __init__(self):
        self.field_size = 10
        self.state = np.zeros([self.field_size, self.field_size], dtype=np.int32)
        # Generate walls at random
        for _ in range(40):
            self.state[np.random.choice(self.field_size), np.random.choice(self.field_size)] = -2
        # Decide the location of the enemy at random
        self.state[np.random.choice(self.field_size), np.random.choice(self.field_size)] = -1

        # Decide the location of the enemy at random
        self.state[np.random.choice(self.field_size), np.random.choice(self.field_size)] = 1

    def inpute_key(self, key):
        loc_enemy = np.array(np.where(self.state == -1)).ravel()
        loc_dqn = np.array(np.where(self.state == 1)).ravel()
        s_t = self.state

        return 0

    def score(self):
        return 0

    def print_board(self):
        for row in self.state:
            for cell in row:
                if cell == 1:
                    print "*",
                elif cell == -1:
                    print "+",
                elif cell == -2:
                    print "#",
                else:
                    print "-",
            else:
                print ""


if __name__ == "__main__":
    em = SampleEmulator()
    em.print_board()
