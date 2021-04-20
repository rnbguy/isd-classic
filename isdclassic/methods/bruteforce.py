import logging
import itertools
import numpy as np
from isdclassic.methods.common import ISDWithoutLists

logger = logging.getLogger(__name__)


class Bruteforce(ISDWithoutLists):
    def __init__(self, h, s, t):
        super().__init__(h, s, t, ISDWithoutLists.ALG_BRUTEFORCE)

    def run(self):
        error = np.zeros(self.n)
        for i in itertools.combinations(range(self.n), self.t):
            # extract only  the columns indexed by i
            h_extr = self.h[:, i]
            # sum the columns by rows
            if np.array_equal(h_extr.sum(axis=1) % 2, self.s):
                for j in i:
                    error[j] = 1
                self.result['indexes'] = i
                return error
