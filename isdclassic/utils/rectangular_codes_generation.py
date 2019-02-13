import numpy as np
import logging

_logger = logging.getLogger(__name__)


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
    The standard way to create a non-systematic parity matrix H for the
    Hamming code [2^m - 1, 2^m - 1 - m] (e.g., if m = 3, we have a [7, 4])
    is to enumerate all the integers b/w 1 and 2^m - 1 in binary form.

    All of this binary integers will represent the column vectors of the
    parity matrix H. So, we put all of them one after the other.
    As for the usual notation, 2^m - 1 = n, 2^m - 1 - m = k, r = n - k
    """
    _assert_hamming(n, k)
    # e.g. n = 7, k = 4, r = 3
    r = n - k
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


def generate_parity_matrix_from_systematic_g(g):
    """
    Given a systematic generator matrix g, returns the corresponding parity matrix h.
    Remember that g = (I_{k x k} | A_{k x r}) and h is obtained as
    h = (A^T_{r x k} | I_{r x r})

    :param g: the generator matrix G
    :returns: the corresponding parity matrix H
    :rtype: np.array (r x n)

    """
    k = g.shape[0]
    n = g.shape[1]
    r = n - k
    id_k = np.eye(k)
    rows = range(0, k)
    cols = range(0, k)
    assert np.array_equal(id_k, g[rows][:, cols]), "Not in systematic form"
    cols = range(k, n)
    a = g[rows][:, cols]
    a_t = a.T
    id_r = np.eye(r)
    h = np.concatenate((a_t, id_r), axis=1)
    return h


def generate_generator_matrix_from_systematic_h(h):
    r = h.shape[0]
    n = h.shape[1]
    k = n - r
    id_k = np.eye(k)
    id_r = np.eye(r)
    rows = range(0, r)
    cols = range(k, n)
    assert np.array_equal(id_r, h[rows][:, cols]), "Not in systematic form"
    cols = range(0, k)
    a_t = h[rows][:, cols]
    a = a_t.T
    g = np.concatenate((id_k, a), axis=1)
    return g


# see ErrorCorrection1.pdf, page 44
# if original h had (n, k, d), the new h will have(n+1, k+1, d*)
# If original d is odd, than d*= d + 1
# However, if systematic=True it's not assured that new d is d+1
def add_overall_parity_bits_to_h(h, systematic=False):
    r = h.shape[0]
    n = h.shape[1]
    k = n - r
    ones_row = np.ones((1, n))
    new_col = np.zeros((r + 1, 1))
    new_col[r] = 1
    h2 = np.concatenate((h, ones_row), axis=0)
    h2 = np.concatenate((h2, new_col), axis=1)
    if systematic:
        h2[r] = (np.sum(h2, axis=0)) % 2
    return h2


# Useless, but useful insights
def generate_parity_matrix_systematic_for_hamming(n, k, d=3):
    """
    Generate a systematic matrix for an [n, k] code. Basically, we first get a
    non-systematic parity matrix H using the appropriate function. Then, we
    permute the columns until the left (n-k)x(n-k) submatrix of H corresponds
    to the identity matrix I.
    WARNING: This can take a very huge time, be careful!
    """
    pass
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
