import logging
import numpy as np
import itertools
from isdclassic.methods.common import ISDWithoutLists

logger = logging.getLogger(__name__)


class LeeBrickell(ISDWithoutLists):
    def __init__(self, h, s, t, p):
        super().__init__(h, s, t, ISDWithoutLists.ALG_LEE_BRICKELL)
        self.p = p
        assert self.k >= self.p, "k should be at least p, while k is {} and p is {}".format(
            self.k, self.p)

    def bruteforce(self, hr, s_sig):
        wanted_sum = self.t - self.p
        logger.debug('s_sig is {}, hrref is \n{}'.format(s_sig, hr))
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
                logger.debug("FOUND!! ")
                e_hat = np.concatenate((np.zeros(self.k), sum_to_s))
                logger.debug("e hat is {}".format(e_hat))
                for j in i:
                    e_hat[j] = 1
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
            e_hat = self.bruteforce(hr, s_sig)
            if e_hat is None: # 
                continue
            else:
                exit_condition = True
        # Double check, but at this point it's always true
        # if np.sum(e_hat) == self.t:
        self.result['e_hat'] = e_hat
        self.result['hr'] = hr
        self.result['perm'] = perm
        self.result['s_sig'] = s_sig
        self.result['u'] = u
        self.result['v'] = hr[:, 0:self.k]
        e = np.mod(np.dot(e_hat, perm.T), 2)
        self.result['completed'] = True
        return e
