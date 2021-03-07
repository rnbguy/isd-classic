import numpy as np
import unittest
from parameterized import parameterized
from test.isd_common import ISDTest
from isdclassic.methods import prange
from isdclassic.utils import rectangular_codes_hardcoded


class ISDPrangeTest(ISDTest):
    @classmethod
    def setUpClass(cls):
        # Just to use prange logger
        super().setUpClass()
        # import logging
        # prange_isd_logger = logging.getLogger('isdclassic.methods.prange')
        # prange_isd_logger.setLevel(cls.logger.level)
        # prange_isd_logger.handlers = cls.logger.handlers

    @parameterized.expand([
        # TRUE
        ("n4_k1_d4_w1", 4, 1, 4, 1, True),
        ("n7_k4_d3_w1", 7, 4, 3, 1, True),
        ("n8_k4_d4_w1", 8, 4, 4, 1, True),
        ("n15_k11_d3_w1", 15, 11, 3, 1, True),
        ("n16_k11_d4_w1", 16, 11, 4, 1, True),
    ])
    def test_simple_h_s_d_w(self, name, n, k, d, w, scramble):
        self.common(name, n, k, d, w, scramble)

    @parameterized.expand([
        ("n23_k12_d7_w3", 23, 12, 7, 3, False),
    ])
    @unittest.skipIf(not ISDTest.SLOW, "Skipped slow test")
    def test_slow_h_s_d_w(self, name, n, k, d, w, scramble):
        self.common(name, n, k, d, w, scramble)

    @parameterized.expand([
        # Quantum reference
        ("n8_k4_d4_w2", 8, 4, 4, 2, True),
        ("n8_k4_d4_w2", 8, 3, 4, 2, True),
        # Slow bcz k low, unlikely to have the condition satisfied
        # However, faster than leon and lee
        ("n8_k3_d4_w2", 8, 3, 4, 2, True),
        ("n8_k2_d5_w3", 8, 2, 5, 3, True),
        ("n8_k1_d7_w3", 8, 1, 7, 3, True),
    ])
    @unittest.skipIf(not ISDTest.FAKE, "Skipped fake test")
    def test_fake_h_s_d_w(self, name, n, k, d, w, scramble):
        self.common(name, n, k, d, w, scramble)

    def common(self, name, n, k, d, w, scramble):
        # first _ is the G, we are not interested in it
        # second _ is the isHamming boolean value, not interested
        h, _, syndromes, errors, w, _ = rectangular_codes_hardcoded.get_isd_systematic_parameters(
            n, k, d, w)
        self.logger.info(
            "Launching TEST w/ n = {0}, k = {1}, d = {2}, w = {3}".format(
                n, k, d, w))
        self.logger.debug("h = \n{0}".format(h))

        syndromes, errors = self.get_max_syndromes_errors(syndromes, errors)
        h_p, errors_p = self.scramble_h_errors(
            h, errors) if scramble else (h, errors)
        # We avoid the iterpermutations (slow) if slow test is off
        max_mode_idx = 3 if ISDTest.SLOW else 2
        for rref_mode in prange.Prange.RREF_MODES[:max_mode_idx]:
            for i, s in enumerate(syndromes):
                with self.subTest(rref_mode=rref_mode, h=h_p, s=s, w=w):
                    self.logger.info("Launching SUBTEST w/ s = {0}".format(s))
                    pra = prange.Prange(h_p, s, w, rref_mode)
                    e = pra.run()
                    self.logger.debug(
                        "For s = {0}, w = 1, h = \n{1}\nerror is {2}".format(
                            s, h_p, e))
                    np.testing.assert_array_almost_equal(e, errors_p[i])


if __name__ == '__main__':
    unittest.main()
