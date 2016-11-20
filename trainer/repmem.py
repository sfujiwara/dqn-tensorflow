# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


COLUMNS = ['s_t', 'a_t', 'r_t', 's_t_plus_1', 'terminal']


class ReplayMemory:

    def __init__(self, memory_size=100000):
        self.df = pd.DataFrame(columns=COLUMNS)
        self.memory_size = memory_size

    # TODO: Restrict memory usage
    def store(self, s_t, a_t, r_t, x_t_plus_1, terminal):
        row = pd.Series(data=[s_t, a_t, r_t, x_t_plus_1, terminal], index=COLUMNS)
        self.df = self.df.append(row, ignore_index=True)
        # Restrict memory usage
        if self.df.size > self.memory_size:
            self.df = self.df.sample(self.memory_size, replace=True)

    def sample(self, size):
        df_mini_batch = self.df.iloc[np.random.choice(self.df.index, size)]
        result = {
            "s_t": np.array(list(df_mini_batch["s_t"])),
            "a_t": np.array(list(df_mini_batch["a_t"])),
            "r_t": np.array(list(df_mini_batch["r_t"])),
            "s_t_plus_1": np.array(list(df_mini_batch["s_t_plus_1"])),
            "terminal": np.array(list(df_mini_batch["terminal"])),
        }
        return result
