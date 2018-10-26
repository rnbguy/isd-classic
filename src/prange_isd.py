import numpy as np
import sympy
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


# Random permutation of columns (by default) or rows
# Return the permutated matrix and the permutation matrix utilized
def permute(m, cols=True):
    axis = m.shape[1] if cols else m.shape[0]
    i = np.eye(axis)
    p = np.random.permutation(i)

    # Post-multiply to permute columns, pre-multiply to permute rows
    mp = np.dot(m, p) if cols else np.dot(p, m)
    return (mp, p)


# Uses LU decomposition w/ Doolittle algorithm, i.e. PA = LU.
# Returns:
# - P, the permutation of rows
# - L, the matrix used to obtain the Reduced Row Echelon Form
# - U, the matrix A in Reduced Row Echelon Form (and also unitary)
def rref(m):
    # TODO use LU functions
    pass


# s is r*1, h is r*n and possibly non systematic, t is weight of the error (i.e. # of ones)
# return error vector e (n*1) s.t. He^ = s AND weight(e) = t
def isd(s, t, h):
    r = h.shape[0]
    n = h.shape[1]
    k = n - r
    _logger.debug("\nr={0}, n={1}, k={2}".format(r, n, k))

    # Random permutation of H to obtain H_hat
    # We can't use numpy because we must keep track the permutation
    i = np.eye(r)
    exit_c1 = False
    exit_c2 = False
    i = 0
    while (not exit_c1):
        while (not exit_c2):
            hp, p = permute(h)
            hr, u = rref(hp)
            w = hr[:, r + 1:]
            i += 1
            if (np.array_equal(w, i)):
                exit_c2 = True
            if (i == 2):
                exit_c2 = True
        exit_c1 = True

    # Get RREF of H using transformation U, i.e. UH = [V | W]
    # where W should be an Identity r*r matrix

    # Apply Us to obtain s_signed

    # e_hat is [0_{1*k} S_signed^transposed]
