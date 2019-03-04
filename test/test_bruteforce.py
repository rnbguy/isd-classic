import numpy as np
import unittest
from parameterized import parameterized
from test.isd_common import ISDTest
from isdclassic.methods import bruteforce
from isdclassic.utils import rectangular_codes_hardcoded


class ISDBruteforceTest(ISDTest):
    @parameterized.expand([
        ("n4_k1_d4_w1", 7, 4, 3, 1, True),
        ("n7_k4_d3_w1", 7, 4, 3, 1, True),
        ("n15_k11_d3_w1", 15, 11, 3, 1, True),
        ("n16_k11_d4_w1", 16, 11, 4, 1, True),
        ("n23_k12_d7_w3", 23, 12, 7, 3, False),
        # Fake, no need to separate
        ("n8_k4_d4_w2", 8, 4, 4, 2, True),
        ("n8_k3_d4_w2", 8, 3, 4, 2, True),
        ("n8_k2_d5_w3", 8, 2, 5, 3, True),
        ("n8_k1_d7_w3", 8, 1, 7, 3, True),
    ])
    def test_h_s_d_w(self, name, n, k, d, w, scramble):
        # first _ is the G, we are not interested in it
        # second _ is the isHamming boolean value, not interested
        h, _, syndromes, errors, w, _ = rectangular_codes_hardcoded.get_isd_systematic_parameters(
            n, k, d, w)
        self.logger.debug("h = \n{0}".format(h))
        syndromes, errors = self.get_max_syndromes_errors(syndromes, errors)
        h_p, errors_p = self.scramble_h_errors(
            h, errors) if scramble else (h, errors)
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
