import numpy as np
import logging

_logger = logging.getLogger(__name__)


def get_isd_parameters(n, k, d, w):
    if (n, k, d, w) == (7, 4, 3, 1):
        return _get_7_4_3_w1()
    elif (n, k, d, w) == (15, 11, 4, 1):
        return _get_15_11_4_w1()
    else:
        raise AttributeError("Combination not supported yet.")


def _get_7_4_3_w1():
    h_systematic = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1],
                             [1, 0, 0], [0, 1, 0], [0, 0, 1]]).T

    syndromes = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                          [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    error_patterns_systematic = np.array([[0, 0, 0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 0, 0],
                                          [0, 1, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 0]])
    # 1 is the weight
    return h_systematic, syndromes, error_patterns_systematic, 1


def _get_15_11_4_w1():
    h = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                  [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]])
    syndromes = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0],
         [0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1],
         [1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]])
    errors = np.array(
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    return h, syndromes, errors, 1


def get_systematic_g(n, k, d):
    if (n, k, d) == (9, 4, 4):
        return _get_944_systematic_g()
    elif (n, k, d) == (7, 4, 3):
        return _get_743_systematic_g()
    else:
        raise AttributeError(
            "[{0}, {1}, {2}] parity matrix not yet implemented".format(
                n, k, d))


def _get_944_systematic_g():
    """
    Get a fixed generator matrix G for the [9, 4, 4] code
    """
    i4 = np.eye(4)
    # i4 = np.eye(4, dtype=np.bool)
    a45 = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1],
                    [1, 1, 1, 1]]).T

    g = np.concatenate((i4, a45), axis=1)
    _logger.debug("G = \n{0}".format(g))
    h_rref, cols = sympy.Matrix(g).rref()
    return g


def _get_743_systematic_g():
    """
    Get a fixed systematic G matrix for a [7, 4, 3] code, i.e. the matrix
    [[1 0 0 0 ], [0 1 0 0], [0 0 1 0], [0 0 0 1], [1 1 0 1], [1 0 1 1], [0 1 1 1]].T
    """
    i4 = np.eye(4)
    a43 = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]).T

    g = np.concatenate((i4, a43), axis=1)
    _logger.debug("***G***\n{0}".format(g))
    return g


def get_codeword_of_message(u, g):
    """
    Given an original message and a generation matrix G, returns the corresponding codeword.

    :param u: the original vector message, 1xk
    :param g: the generation matrix kxn
    :returns: codeword vector 1xk
    :rtype: numpy.array
    """
    # n = g.shape[1]  # columns
    # k = g.shape[0]  # rows
    # r = n - k
    # _logger.debug("n = {0}, k = {1},  r = {2}".format(n, k, r))

    c = np.dot(u, g) % 2
    return c
