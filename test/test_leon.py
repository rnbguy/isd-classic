import numpy as np
import unittest
from parameterized import parameterized
from test.isd_common import ISDTest
from isdclassic.methods import leon
from isdclassic.utils import rectangular_codes_hardcoded


class ISDLeonTest(ISDTest):
    @classmethod
    def setUpClass(cls):
        # Just to use leon logger also
        super().setUpClass()
        import logging
        lee_logger = logging.getLogger('isdclassic.methods.leon')
        lee_logger.setLevel(cls.logger.level)
        lee_logger.handlers = cls.logger.handlers
        return
        lee_logger = logging.getLogger('isdclassic.utils.lu')
        lee_logger.setLevel(cls.logger.level)
        lee_logger.handlers = cls.logger.handlers

    @parameterized.expand([
        # FAKE
        # ("n8_k2_d5_w3_p1_l1", 8, 2, 5, 3, 1, 1, True),
        # ("n8_k2_d5_w3_p1_l2", 8, 2, 5, 3, 1, 2, True),
        # ("n8_k2_d5_w3_p1_l3", 8, 2, 5, 3, 1, 3, True),
        # ("n8_k2_d5_w3_p2_l1", 8, 2, 5, 3, 2, 1, True),
        # ("n8_k2_d5_w3_p2_l2", 8, 2, 5, 3, 2, 2, True),
        # ("n8_k2_d5_w3_p2_l3", 8, 2, 5, 3, 2, 3, True),
        # ("n8_k1_d7_w3_p1_l1", 8, 1, 7, 3, 1, 1, True),
        # ("n8_k3_d4_w2_p1_l1", 8, 3, 4, 2, 1, 1, True),
        # ("n8_k3_d4_w2_p1_l2", 8, 3, 4, 2, 1, 2, True),
        # ("n8_k3_d4_w2_p1_l3", 8, 3, 4, 2, 1, 3, True),
        ("n8_k4_d4_w2_p1_l1", 8, 4, 4, 2, 1, 1, True),
        ("n8_k4_d4_w2_p1_l2", 8, 4, 4, 2, 1, 2, True),
        ("n8_k4_d4_w2_p1_l3", 8, 4, 4, 2, 1, 3, True),
        ("n8_k4_d4_w2_p2_l1", 8, 4, 4, 2, 2, 1, True),
        ("n8_k4_d4_w2_p2_l2", 8, 4, 4, 2, 2, 2, True),
        ("n8_k4_d4_w2_p2_l3", 8, 4, 4, 2, 2, 3, True),
        # TRUE
        ("n4_k1_d4_w1_p1_l1", 4, 1, 4, 1, 1, 1, True),
        ("n4_k1_d4_w1_p1_l2", 4, 1, 4, 1, 1, 2, True),
        ("n4_k1_d4_w1_p1_l3", 4, 1, 4, 1, 1, 3, True),
        ("n7_k4_d3_w1_p1_l1", 7, 4, 3, 1, 1, 1, True),
        ("n7_k4_d3_w1_p1_l2", 7, 4, 3, 1, 1, 2, True),
        ("n7_k4_d3_w1_p1_l3", 7, 4, 3, 1, 1, 3, True),
        ("n8_k4_d4_w1_p1_l2", 8, 4, 4, 1, 1, 2, True),
        ("n8_k4_d4_w1_p1_l3", 8, 4, 4, 1, 1, 3, True),
        ("n15_k11_d3_w1_p1_l1", 15, 11, 3, 1, 1, 1, True),
        ("n15_k11_d3_w1_p1_l2", 15, 11, 3, 1, 1, 2, True),
        ("n15_k11_d3_w1_p1_l3", 15, 11, 3, 1, 1, 3, True),
        ("n15_k11_d3_w1_p1_l4", 15, 11, 3, 1, 1, 4, True),
        ("n23_k12_d7_w3_p1_l2", 23, 12, 7, 3, 1, 2, True),
        ("n23_k12_d7_w3_p1_l3", 23, 12, 7, 3, 1, 3, True),
        ("n23_k12_d7_w3_p2_l2", 23, 12, 7, 3, 2, 2, True),
        ("n23_k12_d7_w3_p2_l3", 23, 12, 7, 3, 2, 3, True),
        ("n23_k12_d7_w3_p3_l1", 23, 12, 7, 3, 3, 1, True),
        ("n23_k12_d7_w3_p3_l2", 23, 12, 7, 3, 3, 2, True),
    ])
    def test_h_s_d_w_p_l(self, name, n, k, d, w, p, l, scramble):
        # first _ is the G, we are not interested in it
        # second _ is the isHamming boolean value, not interested
        h, _, syndromes, errors, w, _ = rectangular_codes_hardcoded.get_isd_systematic_parameters(
            n, k, d, w)
        self.logger.info(
            "Launching TEST w/ n = {0}, k = {1}, d = {2}, w = {3}, p = {4}, l = {5}"
            .format(n, k, d, w, p, l))
        self.logger.debug("h = \n{0}".format(h))

        syndromes, errors = self.get_max_syndromes_errors(syndromes, errors)
        h_p, errors_p = self.scramble_h_errors(
            h, errors) if scramble else (h, errors)
        for i, s in enumerate(syndromes):
            with self.subTest(h=h_p, s=s, w=w, p=p, l=l):
                self.logger.info("Launching SUBTEST w/ s = {0}".format(s))
                leo = leon.Leon(h_p, s, w, p, l)
                e = leo.run()
                self.logger.debug(
                    "For s = {0}, w = {1}, p = {2}, l = {5} h = \n{3}\nerror is {4}"
                    .format(s, w, p, h_p, e, l))
                np.testing.assert_array_almost_equal(e, errors_p[i])


if __name__ == '__main__':
    unittest.main()
