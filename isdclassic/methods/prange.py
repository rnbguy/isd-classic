import logging
import numpy as np
from isdclassic.methods.common import ISDWithoutLists

logger = logging.getLogger(__name__)


class Prange(ISDWithoutLists):
    def __init__(self, h, s, t):
        super().__init__(h, s, t, ISDWithoutLists.ALG_PRANGE)

    def run(self):
        """Run the isd algorithm
        
        :param s: the (n-k)x1 syndrome vector
        :param t: the weight of the error (i.e. the number of ones)
        :param h: the parity matrix, possibly in nonsystematic form
        :returns: the nx1 error vector s.t. He.T = s AND weight(e) = t
        :rtype: numpy.array

        """

        # From now on exit_condition_weight is used to continue the algorithm until we found
        # the right weight for the error
        exit_condition_weight = False
        while (not exit_condition_weight):
            hr, u, perm, s_sig = self.get_matrix_rref()
            t_hat = np.sum(s_sig)
            logger.debug("Weight of s is {0}".format(t_hat))
            exit_condition_weight = t_hat == self.t
            if exit_condition_weight:
                logger.debug("WEIGHT IS CORRECT, FOUND e")
                # e_hat is the concatenation of all zeros 1xk vector and s_signed^transposed
                e_hat = np.concatenate([np.zeros(self.k), s_sig.T])
                logger.info("s signed is {0}".format(s_sig))
                logger.info("e hat is {0}".format(e_hat))
                logger.info("perm is \n{0}".format(perm))
                logger.info("u is \n{0}".format(u))
                logger.info("hr, that is u.h.p is \n{0}".format(hr))
                self.result['hr'] = hr
                self.result['perm'] = perm
                self.result['s_sign'] = s_sig
                self.result['u'] = u
                self.result['e_hat'] = e_hat
            else:
                logger.debug("Weight is wrong, retrying")

        # return the error vector multiplying e_hat by the permutation matrix
        e = np.mod(np.dot(e_hat, perm.T), 2)
        self.result['completed'] == True
        logger.info("s was {0}, e is {1}".format(self.s, e))
        return e
