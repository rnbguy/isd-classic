import numpy as np
import sympy
import rectangular_code
import logging


def _assert_hamming(n, k, d=3):
    r = n - k
    assert_msg = "[{0}, {1}] is not an Hamming code".format(n, k)
    assert r >= 1, assert_msg
    assert d <= r, assert_msg
    assert n == 2**r - 1, assert_msg


def generate_parity_matrix_nonsystematic_for_hamming(n, k, d=3):
    """
    This only works for Hamming codes, i.e. codes in the form
    [2^m - 1, 2^m - 1 - m]
    which always have a minimum Hamming distance of 3.
    The standard way to create a non-systematix parity matrix H for the
    Hamming code [2^m - 1, 2^m - 1 - m] (e.g., if m = 3, we have a [7, 4])
    is to enumerate all the integers b/w 1 and 2^m in binary form. All of this
    binary integers will represent a column vector of the parity matrix H.
    So, we put all of them one after the other.
    As for the usual notation, 2^m - 1 = n and 2^m - 1 - m =k.
    """
    _assert_hamming(n, k)
    # e.g. n = 7, k = 4, r = 3
    support = np.array([i for i in range(1, n + 1)])
    _logger.debug("SUPPORT ARRAY\n{0}".format(support))
    # Create columns from previous range.
    # F.e. if n = 7, r = 3 this will create COLUMNS
    # [0 0 1], [0 1 0], [0 1 1], [1 0 0], [1 0 1], [1 1 0], [1 1 1]
    # Then it put the columns together in a 3x7 (rxk) matrix
    # This is the standard way of creating a nonsystematic parity matrix
    # for an hamming code
    h = (((support[:, None] & (1 << np.arange(r)))) > 0).T.astype(int)
    _logger.debug("H\n{0}".format(h))
    return h


def generate_parity_matrix_systematic_for_hamming(n, k, d=3):
    """
    Obtain a systematic matrix for an [n, k] code. Basically, we first get a
    non-systematix parity matrix H using the appropriate function. Then, we
    permute the columns until the left (n-k)x(n-k) submatrix of H corresponds
    to the identity matrix I.
    WARNING: This can take a very huge time, be careful!
    """
    _assert_hamming(n, k, d)
    r = n - k
    h = generate_parity_matrix_nonsystematic(n, k, d)

    ir = np.eye(r)
    i = 0
    exit_condition = False
    while (not exit_condition):
        _logger.debug("Random permutation number {0}".format(i))

        # Numpy Random permutation works on rows, we have to permute columns
        # TODO we generate random columns to permute, but this really should
        # be an educated work to avoid repeating same permutation over and over
        hp = np.random.permutation(h.T).T

        # Extract the right submatrix r*r
        # Slice all the rows, : , while columns from r-1 on
        # w = hpr[:, r - 1:]
        w = hp[:, r + 1:]
        _logger.debug("\nHP=\n{0}\nW =\n{1}".format(hp, w))

        i += 1
        exit_condition = np.array_equal(w, ir)  # or i > 100
    return hp
