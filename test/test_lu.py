import numpy as np
import unittest
from parameterized import parameterized
from test.isd_common import ISDTest
from isdclassic.utils import lu
from isdclassic.utils.rectangular_codes_generation import generate_parity_matrix_nonsystematic_for_hamming


class TestLPU(ISDTest):
    @parameterized.expand([
        ["7_4_False", 7, 4, False],
        ["7_4_True", 7, 4, True],
        ["15_11_False", 15, 11, False],
        ["15_11_True", 15, 11, True],
    ])
    def test_parity_matrix(self, name, n, k, startAtEnd):
        h = generate_parity_matrix_nonsystematic_for_hamming(n, k)
        # Sum of ones of H, just a quick way to check that all the transformations
        # applied doesn't modify the overall structure of H
        sum_h = np.sum(h)
        self.logger.debug(
            "TESTING PARITY MATRIX (n = {0}, k = {1}), startAtEnd = {2},\nH = \n{3}"
            .format(n, k, startAtEnd, h))
        self.logger.debug("Overall matrix sum is {0}".format(sum_h))

        # h_rref is the original H in rref, l is the transformation matrix applied
        # overall to obtain u from H
        _, l, h_rref = lu.get_rref(h, 3, startAtEnd=startAtEnd, mod=2)
        # The idea is that if all items are None, then the algorithm wasn't able to
        # find a proper unitary rref submatrix, so we shuffle again the columns
        # of the original matrix
        while (all(item is None for item in (l, h_rref))):
            h = np.random.permutation(h.T).T
            self.logger.debug("PERMUTED, new H = \n{0}".format(h))
            self.logger.debug(
                "Overall matrix H sum (number of ones) is {0}".format(
                    np.sum(h)))
            _, l, h_rref = lu.get_rref(h, 3, startAtEnd=startAtEnd, mod=2)

        # L*H matrix, should be equal to the original matrix if all went good
        self.logger.debug("H=\n{0}\nH_RREF=\n{1}\nLTOT=\n{2}\n".format(
            h, h_rref, l))
        self.logger.debug("Overall matrix sum is {0}".format(np.sum(h_rref)))
        np.testing.assert_equal(sum_h, np.sum(h_rref))

        lh = np.mod(np.dot(l, h), 2)
        np.testing.assert_almost_equal(lh, h_rref)

        id = np.eye(n - k)
        square_matrix_start = 0 if not startAtEnd else k
        square_matrix_end = n - k if not startAtEnd else n
        self.logger.debug(
            "Slicing with startAtEnd={2}, start={0}, end={1}".format(
                square_matrix_start, square_matrix_end, startAtEnd))

        # Take the right-most n-k square matrix, i.e.:
        # - all the rows (:)
        # - columns from start to end
        tst = h_rref[:, square_matrix_start:square_matrix_end]
        np.testing.assert_almost_equal(tst, id)

    def test_zero(self):
        a = np.array([[0, 1, 1], [3, 1, 1], [8, 7, 9]])
        exp_res = np.eye(3)
        p, l, u = lu.get_rref(a, 3)
        lta = np.dot(l, a)

        self.logger.debug(
            "A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}".format(
                a, u, l, lta))
        np.testing.assert_almost_equal(u, exp_res)
        np.testing.assert_almost_equal(lta, u)

    def test_one(self):
        m = np.array([[1, 0, 1, 6], [0, -3, 1, 7], [2, 1, 3, 15]])
        exp_res = np.eye(3)
        tmp = np.array([[2, -1, 4]]).T
        exp_res = np.hstack((exp_res, tmp))
        p, l, u = lu.get_rref(m, 3)
        ltm = np.dot(l, m)

        self.logger.debug(
            "M=\n{0}\nM_RREF=\n{1}\nLTOT=\n{2}\nLTOT*M=\n{3}".format(
                m, u, l, ltm))
        np.testing.assert_almost_equal(u, exp_res)
        np.testing.assert_almost_equal(ltm, u)

    def test_pdf_stopat1(self):
        a = np.array([[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]])
        exp_res = np.array([[2, 1, 1, 0], [0, 1, 1, 1], [0, 0, 2, 2],
                            [0, 0, 0, 2]])
        p, l, u = lu.get_rref(a, 1)
        tst = np.dot(l, a)
        self.logger.debug(
            "A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}".format(
                a, u, l, tst))

        np.testing.assert_almost_equal(u, exp_res)
        np.testing.assert_almost_equal(tst, u)

    def test_pdf_stopat3(self):
        a = np.array([[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]])
        exp_res = np.array([[2, 1, 1, 0], [0, 1, 1, 1], [0, 0, 2, 2],
                            [0, 0, 0, 2]])
        p, l, u = lu.get_rref(a, 1)
        tst = np.dot(l, a)
        self.logger.debug(
            "A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}".format(
                a, u, l, tst))

        p, l, u = lu.get_rref(a, 3)
        exp_res = np.eye(4)
        tst = np.dot(l, a)
        self.logger.debug(
            "A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}".format(
                a, u, l, tst))

        np.testing.assert_almost_equal(u, exp_res)
        np.testing.assert_almost_equal(tst, u)

    def test_wiki(self):
        a = np.array([
            [2, 1, -1, 8],
            [-3, -1, 2, -11],
            [-2, 1, 2, -3],
        ])
        exp_res = np.array([
            [1, 0, 0, 2],
            [0, 1, 0, 3],
            [0, 0, 1, -1],
        ])

        ptot, ltot, a_rref = lu.get_rref(a)
        lta = np.dot(ltot, a)

        self.logger.debug(
            "A = \n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}\nPTOT=\n{4}".
            format(a, a_rref, ltot, lta, ptot))
        np.testing.assert_almost_equal(a_rref, exp_res)
        np.testing.assert_almost_equal(lta, a_rref)


if __name__ == "__main__":
    unittest.main()
