import numpy as np
import logging
import utils
import hamming_codes as hc


def test_parity_matrix(n, k, startAtEnd=False):
    h = hc.get_parity_matrix_nonsystematic(n, k)
    # Sum of ones of H, just a quick way to check that all the transformations
    # applied doesn't modify the overall structure of H
    sum_h = np.sum(h)
    _logger.debug(
        "TESTING PARITY MATRIX (n = {0}, k = {1}), startAtEnd = {2},\nH = \n{3}"
        .format(n, k, startAtEnd, h))
    _logger.debug("Overall matrix sum is {0}".format(sum_h))

    # h_rref is the original H in rref, l is the transformation matrix applied
    # overall to obtain u from H
    _, l, h_rref = utils.get_rref(h, 3, startAtEnd=startAtEnd, mod=2)
    # The idea is that if all items are None, then the algorithm wasn't able to
    # find a proper unitary rref submatrix, so we shuffle again the columns
    # of the original matrix
    while (all(item is None for item in (l, h_rref))):
        h = np.random.permutation(h.T).T
        _logger.debug("*********** PERMUTED, new H = \n{0}".format(h))
        _logger.debug("Overall matrix H sum (number of ones) is {0}".format(
            np.sum(h)))
        _, l, h_rref = utils.get_rref(h, 3, startAtEnd=startAtEnd, mod=2)

    # L*H matrix, should be equal to the original matrix if all went good
    lh = np.mod(np.dot(l, h), 2)
    _logger.debug("H=\n{0}\nH_RREF=\n{1}\nLTOT=\n{2}\n".format(h, h_rref, l))
    _logger.debug("Overall matrix sum is {0}".format(np.sum(h_rref)))

    id = np.eye(n - k)
    square_matrix_start = 0 if not startAtEnd else k
    square_matrix_end = n - k if not startAtEnd else n
    _logger.debug("Slicing with startAtEnd={2}, start={0}, end={1}".format(
        square_matrix_start, square_matrix_end, startAtEnd))

    # Take the right-most n-k square matrix, i.e.:
    # - all the rows (:)
    # - columns from start to end
    tst = h_rref[:, square_matrix_start:square_matrix_end]
    _logger.debug("Asserting L * H, should be equal to H_RREF")
    np.testing.assert_almost_equal(tst, id)
    _logger.debug("...OK")
    _logger.debug("Asserting L * H, should be equal to U")
    np.testing.assert_almost_equal(lh, h_rref)
    _logger.debug("...OK")
    _logger.debug("Asserting number of ones in h and u is equal".format(lh))
    np.testing.assert_equal(sum_h, np.sum(h_rref))


def test_zero():
    _logger.debug("***test zero")
    a = np.array([[0, 1, 1], [3, 1, 1], [8, 7, 9]])
    exp_res = np.eye(3)
    p, l, u = utils.get_rref(a, 3)
    lta = np.dot(l, a)

    _logger.debug("A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}".format(
        a, u, l, lta))
    _logger.debug("Asserting equalities...")
    np.testing.assert_almost_equal(u, exp_res)
    np.testing.assert_almost_equal(lta, u)
    _logger.debug("...OK")


def test_one():
    _logger.debug("***test one")
    m = np.array([[1, 0, 1, 6], [0, -3, 1, 7], [2, 1, 3, 15]])
    exp_res = np.eye(3)
    tmp = np.array([[2, -1, 4]]).T
    exp_res = np.hstack((exp_res, tmp))
    p, l, u = utils.get_rref(m, 3)
    ltm = np.dot(l, m)

    _logger.debug("M=\n{0}\nM_RREF=\n{1}\nLTOT=\n{2}\nLTOT*M=\n{3}".format(
        m, u, l, ltm))
    _logger.debug("Asserting equalities...")
    np.testing.assert_almost_equal(u, exp_res)
    np.testing.assert_almost_equal(ltm, u)
    _logger.debug("...OK")


def test_pdf():
    _logger.debug("***test pdf")
    _logger.debug("***stopAt 1***")
    a = np.array([[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]])
    exp_res = np.array([[2, 1, 1, 0], [0, 1, 1, 1], [0, 0, 2, 2], [0, 0, 0,
                                                                   2]])
    p, l, u = utils.get_rref(a, 1)
    tst = np.dot(l, a)
    _logger.debug("A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}".format(
        a, u, l, tst))

    _logger.debug("Asserting equalities for stopAt 1...")
    np.testing.assert_almost_equal(u, exp_res)
    np.testing.assert_almost_equal(tst, u)
    _logger.debug("...OK")

    _logger.debug("***stopAt 3***")
    p, l, u = utils.get_rref(a, 3)
    exp_res = np.eye(4)
    tst = np.dot(l, a)
    _logger.debug("A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}".format(
        a, u, l, tst))

    _logger.debug("Asserting equalities for stopAt 1...")
    np.testing.assert_almost_equal(u, exp_res)
    np.testing.assert_almost_equal(tst, u)
    _logger.debug("...ok")


def test_wiki(startAtEnd=False):
    _logger.debug("***test wiki")
    a = np.array([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
    exp_res = np.array([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, -1]])

    ptot, ltot, u = utils.get_rref(a, startAtEnd=startAtEnd)
    lta = np.dot(ltot, a)

    _logger.debug("A = \n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nLTOT*A=\n{3}".format(
        a, u, ltot, lta))
    _logger.debug("Asserting equalities...")
    np.testing.assert_almost_equal(u, exp_res)
    np.testing.assert_almost_equal(lta, u)
    _logger.debug("...OK")


def main():
    test_wiki()
    # test_wiki(True)
    test_pdf()
    test_zero()
    test_one()
    test_parity_matrix(7, 4)
    test_parity_matrix(7, 4, True)
    test_parity_matrix(15, 11)
    test_parity_matrix(15, 11, True)


_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
_handler.setFormatter(_formatter)
if (_logger.hasHandlers()):
    _logger.handlers.clear()
_logger.addHandler(_handler)
_logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    main()
