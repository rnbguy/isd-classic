import itertools
import numpy as np
import logging

logger = logging.getLogger(__name__)


def run(h, s, t):
    logger.debug("s={0}, t={1}, H=\n{2}".format(s, t, h))
    r = h.shape[0]
    n = h.shape[1]
    k = n - r
    logger.debug("r={0}, n={1}, k={2}".format(r, n, k))

    error = np.zeros(n)
    for i in itertools.combinations(range(n), t):
        # extract only the columns indexed by i
        h_extr = h[:, i]
        # sum the columns by rows
        if np.array_equal(h_extr.sum(axis=1) % 2, s):
            for j in i:
                error[j] = 1
            return error
