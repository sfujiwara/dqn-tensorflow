# -*- coding: utf-8 -*-

import numpy as np


class ReplayMemory:

    def __init__(self):
        self.xs_t = []
        self.as_t = []
        self.rs_t = []
        self.xs_t_plus_1 = []
        self.terminals = []

    # TODO: メモリ使用量を調整する
    def store(self, x_t, a_t, r_t, x_t_plus_1, terminal):
        self.xs_t.append(x_t)
        self.as_t.append(a_t)
        self.rs_t.append(r_t)
        self.xs_t_plus_1.append(x_t_plus_1)
        self.terminals.append(terminal)

    def sample(self, size):
        size = min(size, len(self.as_t))
        ind = np.random.choice(len(self.as_t), size=size, replace=False)
        result = {
            "s_t": np.array(self.xs_t)[ind],
            "a_t": np.array(self.as_t)[ind],
            "r_t": np.array(self.rs_t)[ind],
            "s_t_plus_1": np.array(self.xs_t_plus_1)[ind],
            "terminal": np.array(self.terminals)[ind],
        }
        return result

