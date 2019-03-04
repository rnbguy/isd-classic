import unittest
import numpy as np
from parameterized import parameterized
from test.isd_common import ISDTest
import isdclassic.utils.rectangular_codes_hardcoded as rch


class TestHardcoded(ISDTest):
    @parameterized.expand([
        ("4_1_4", 4, 1, 4),
        ("7_4_3", 7, 4, 3),
        ("8_4_4", 8, 4, 4),
        ("9_4_4", 9, 4, 4),
        ("15_11_3", 15, 11, 3),
        ("16_11_4", 16, 11, 4),
        ("23_12_7", 23, 12, 7),
    ])
    def test_g_ht_is_zero(self, name, n, k, d):
        g = rch.get_systematic_g(n, k, d)
        h = rch.get_systematic_h(n, k, d)
        self.assertEqual(0, np.sum(np.dot(g, h.T)) % 2)


if __name__ == "__main__":
    unittest.main()
