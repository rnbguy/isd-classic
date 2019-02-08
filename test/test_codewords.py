import unittest
import numpy as np
from parameterized import parameterized
from test.isd_common import ISDTest
from isd.utils.rectangular_codes_hardcoded import get_systematic_g
from isd.utils.rectangular_codes_compute import get_codeword_of_message


class TestCode(ISDTest):
    @parameterized.expand([
        (7, 4, 3, [1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0]),
        (9, 4, 4, [1, 0, 0, 1], [1, 0, 0, 1, 1, 1, 1, 1, 0]),
        (9, 4, 4, [1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0]),
    ])
    def test_codeword(self, n, k, d, u, c_exp):
        u_a = np.asarray(u)
        c_exp_a = np.asarray(c_exp)
        g = get_systematic_g(n, k, d)
        c = get_codeword_of_message(u_a, g)
        self.logger.debug("Original message is {0}".format(u))
        self.logger.debug("Generated codeword is {0}".format(c))
        np.testing.assert_array_almost_equal(c, c_exp_a)


if __name__ == "__main__":
    unittest.main()
