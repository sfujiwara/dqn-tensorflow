# -*- coding: utf-8 -*-

import collections
import random
import numpy as np


class ReplayMemory:

    def __init__(self, memory_size=2000):
        self.memory = collections.deque(maxlen=memory_size)
        self.memory_size = memory_size

    def store(self, s_t, a_t, r_t, x_t_plus_1, terminal):
        self.memory.append((s_t, a_t, r_t, x_t_plus_1, terminal))

    def sample(self, size):
        mini_batch_size = min(size, len(self.memory))
        mini_batch = zip(*random.sample(self.memory, mini_batch_size))
        mini_batch = [np.array(i) for i in mini_batch]
        return mini_batch
