import logging
import numpy as np
import itertools
from isdclassic.methods.common import ISDWithoutLists

logger = logging.getLogger(__name__)


class Leon(ISDWithoutLists):
    def __init__(self, h, s, t, p, l):
        super().__init__(h, s, t, ISDWithoutLists.ALG_LEON)
        self.p = p
        self.l = l
        assert self.p >= 0 and self.p <= self.t, "p should be between 0 and t, while p is {}".format(
            p)
        assert self.k >= self.p, "k should be at least p, while k is {} and p is {}".format(
            self.k, self.p)
        #TODO assert on l
        assert self.l >= 0 and self.l <= self.r, "l should be between 0 and r, while r is {}".format(
            l)

    def bruteforce(self, hr, s_sig):
        wanted_sum_up = 0
        wanted_sum_down = self.t - self.p
        logger.debug('s_sig is {}, hrref is \n{}'.format(s_sig, hr))
        for i in itertools.combinations(range(self.k), self.p):
            logger.debug("i is {}".format(i))
            # extract only the columns indexed by i, matrix to ensure that it is a matrix
            # and not an array
            h_extr = hr[:, i]
            logger.debug("h extr is {}".format(h_extr))
            # sum the columns by rows
            h_extr_up = h_extr[:self.l, :]
            logger.debug("h extr up is {}".format(h_extr_up))
            sum_to_s_up = (h_extr_up.sum(axis=1) + s_sig[:self.l]) % 2
            logger.debug("sum to s up is {}".format(sum_to_s_up))
            sum_to_s_up_w = np.sum(sum_to_s_up)
            logger.debug("sum to s up weight is {}".format(sum_to_s_up_w))
            # return e_hat
            # if sum_to_s_w == wanted_sum:
            if sum_to_s_up_w == wanted_sum_up:
                h_extr_down = h_extr[self.l:, :]
                logger.debug("h extr down is {}".format(h_extr_down))
                sum_to_s_down = (h_extr_down.sum(axis=1) + s_sig[self.l:]) % 2
                logger.debug("sum to s down is {}".format(sum_to_s_down))
                sum_to_s_down_w = np.sum(sum_to_s_down)
                logger.debug(
                    "sum to s down weight is {}".format(sum_to_s_down_w))
                if sum_to_s_down_w == wanted_sum_down:
                    e_hat = np.concatenate((np.zeros(self.k), sum_to_s_up,
                                            sum_to_s_down))
                    logger.debug("e hat before for is {}".format(e_hat))
                    for j in i:
                        e_hat[j] = 1
                    self.result['indexes'] = i
                    logger.debug("e_hat final is {}".format(e_hat))
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
            if e_hat is None:  #
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
