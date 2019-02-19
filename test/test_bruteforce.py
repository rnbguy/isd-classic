import numpy as np
import unittest
from parameterized import parameterized
from test.isd_common import ISDTest
from isdclassic.methods import bruteforce
from isdclassic.utils import rectangular_codes_hardcoded


class ISDBruteforceTest(ISDTest):
    @parameterized.expand([
        ("n7_k4_d3_w1", 7, 4, 3, 1, True),
        ("n15_k11_d4_w1", 15, 11, 4, 1, True),
        ("n7_k4_d3_w1", 7, 4, 3, 1, False),
    ])
    def test_h_s_d_w(self, name, n, k, d, w, scramble):
        # first _ is the G, we are not interested in it
        # second _ is the isHamming boolean value, not interested
        h, _, syndromes, errors, w, _ = rectangular_codes_hardcoded.get_isd_systematic_parameters(
            n, k, d, w)
        self.logger.debug("h = \n{0}".format(h))
        if (scramble):
            p = np.random.permutation(np.eye(h.shape[1]))
            h_p = np.dot(h, p)
            errors_p = np.dot(errors, p)
        else:
            h_p = h
            errors_p = errors
        for i, s in enumerate(syndromes):
            with self.subTest(h=h_p, s=s, w=w):
                self.logger.debug("Launching prange with s = {0}".format(s))
                bru = bruteforce.Bruteforce(h_p, s, w)
                e = bru.run()
                self.logger.debug(
                    "For s = {0}, w = 1, h = \n{1}\nerror is {2}".format(
                        s, h_p, e))
                np.testing.assert_array_almost_equal(e, errors_p[i])


if __name__ == '__main__':
    unittest.main()
