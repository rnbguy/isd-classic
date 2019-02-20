import logging
import numpy as np
import itertools
from isdclassic.methods.common import ISDWithoutLists

logger = logging.getLogger(__name__)


class LeeBrickell(ISDWithoutLists):
    def __init__(self, h, s, t, p):
        super().__init__(h, s, t, ISDWithoutLists.ALG_LEE_BRICKELL)
        self.p = p

    def bruteforce(self, hr, s_sig):
        wanted_sum = self.t - self.p
        logger.debug("Wanted sum is {}".format(wanted_sum))
        for i in itertools.combinations(range(self.k), self.p):
            logger.debug("i is {}".format(i))
            # extract only the columns indexed by i
            h_extr = hr[:, i]
            # sum the columns by rows
            sum_to_s = (h_extr.sum(axis=1) + s_sig) % 2
            logger.debug("sum to s is {}".format(sum_to_s))
            sum_to_s_w = np.sum(sum_to_s)
            logger.debug("sum to s w is {}".format(sum_to_s_w))
            # return e_hat
            if sum_to_s_w == wanted_sum:
                e_hat = np.concatenate((np.zeros(self.k), sum_to_s))
                logger.debug("e hat is {}".format(e_hat))
                logger.debug("FOUND!! ")
                for j in i:
                    # a = [0] * (j - 1)
                    # b = [1]
                    # c = [0] * (n - len(a) - 1)
                    # print(j)
                    # print(a)
                    # print(b)
                    # print(c)
                    # # e_hat += np.concatenate(([0] * (j - 1), [1], [0] * (n - j)))
                    # e_hat += np.concatenate((a, b, c))
                    # e_hat %= 2
                    e_hat[j] = 1
                self.result['e_hat'] = e_hat
                self.result['v'] = h_extr
                self.result['indexes'] = i
                logger.debug("e_hat is {}".format(e_hat))
                return e_hat

    def run(self):
        """Run the isd algorithm
        
        :param s: the (n-k)x1 syndrome vector
        :param t: the weight of the error (i.e. the number of ones)
        :param h: the parity matrix, possibly in nonsystematic form
        :returns: the nx1 error vector s.t. He.T = s AND weight(e) = t
        :rtype: numpy.array
        
        """
        exit_condition = False
        while (not exit_condition):
            hr, u, perm, s_sig = self.get_matrix_rref()
            self.result['hr'] = hr
            self.result['perm'] = perm
            self.result['s_sign'] = s_sig
            self.result['u'] = u
            e_hat = self.bruteforce(hr, s_sig)
            if np.sum(e_hat) == self.t:
                exit_condition = True
        e = np.mod(np.dot(e_hat, perm.T), 2)
        self.result['completed'] = True
        return e
