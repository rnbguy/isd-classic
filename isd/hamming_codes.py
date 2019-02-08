import numpy as np
import sympy
import rectangular_code
import logging

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
_handler.setFormatter(_formatter)
if (_logger.hasHandlers()):
    _logger.handlers.clear()
_logger.addHandler(_handler)
_logger.setLevel(logging.DEBUG)


def get_parity_matrix_systematic(n, k):
    """
    Obtain a systematic matrix for an [n, k] code. Basically, we first get a
    non-systematix parity matrix H using the appropriate function. Then, we
    permute the columns until the left (n-k)x(n-k) submatrix of H corresponds
    to the identity matrix I.
    """
    r = n - k
    h = get_parity_matrix_nonsystematic(n, k)

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


def get_parity_matrix_nonsystematic(n, k):
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
    # e.g. n = 7, k = 4, r = 3
    r = n - k
    support = np.array([i for i in range(1, n + 1)])
    _logger.debug("***SUPPORT ARRAY***\n{0}".format(support))
    # Create columns from previous range.
    # F.e. if n = 7, r = 3 this will create COLUMNS
    # [0 0 1], [0 1 0], [0 1 1], [1 0 0], [1 0 1], [1 1 0], [1 1 1]
    # Then it put the columns together in a 3x7 (rxk) matrix
    # This is the standard way of creating a nonsystematic parity matrix
    # for an hamming code
    h = (((support[:, None] & (1 << np.arange(r)))) > 0).T.astype(int)
    _logger.debug("***H***\n{0}".format(h))
    return h


def get_fixed_hamming_generator_743():
    """
    Get a fixed systematic G matrix for a [7, 4, 3] code, i.e. the matrix
    [[1 0 0 0 ], [0 1 0 0], [0 0 1 0], [0 0 0 1], [1 1 0 1], [1 0 1 1], [0 1 1 1]].T
    """
    i4 = np.eye(4)
    a43 = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]).T

    g = np.concatenate((i4, a43), axis=1)
    _logger.debug("***G***\n{0}".format(g))
    return g
    pass


# TODO Recheck, not sure it works
# Works only for Hamming codes of the form (2^d - 1, 2^d - 1 - d, d)
# def get_parity_matrix_systematic_maybe(n, k, d):
#     h = get_parity_matrix_nonsystematic(n, k)
#     # h_rref is the Reduced Row Echelon Form
#     # cols is the index of the _independent_ columns.
#     # In our case, len(cols) = m (or d)
#     h_rref, cols = sympy.Matrix(h).rref()

#     # Convert back simpy matrix to numpy
#     h_rref = np.array(h_rref).astype(np.int8)
#     _logger.debug("cols={1}, h_href\n{0}".format(h_rref, cols))

#     # Basically the idea is that the independent columns should be the ones in the identity matrix
#     # (left part of systematic H)
#     # F.e.
#     # 1. if (n=7, k=4) - and so d=3 -
#     # 2. if cols = (0, 1, 3)
#     # we have to put this columns in position (4, 5, 6)

#     # First try:
#     # This assumes that cols[0] contains a (the only) 1 in row[0],
#     # cols[1] contains a 1 in row[1], ecc. This is not necessary the case, so
#     # we should check
#     # TODO: Check for previous
#     for idx, val in enumerate(cols):
#         _logger.debug("idx={0}, val={1}".format(idx, val))
#         h_rref[:, [val, k + idx]] = h_rref[:, [k + idx, val]]
#     # At this point we should have a systematic H, i.e.
#     # H = [A^ | I], w/ A^ = (r*k), I = (r*r)

#     _logger.debug("Systematic form is\n{0}".format(h_rref))
#     return h_rref
