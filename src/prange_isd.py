import numpy as np
import logging
import utils

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
_handler.setFormatter(_formatter)
if (_logger.hasHandlers()):
    _logger.handlers.clear()
_logger.addHandler(_handler)
_logger.setLevel(logging.ERROR)


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
# - MR, the matrix A in Reduced Row Echelon Form (and also unitary)
# - U, the matrix used to obtain the Reduced Row Echelon Form
def rref(m):
    # WARNING: The (ptot, ltot, u) returned from get_rref() are different from ours.
    # Basically, u is the reduced matrix (so u -> mr), ltot is the matrix of transformations
    # applied to it to be reduced (so l -> u)
    _, u, mr = utils.get_rref(m, startAtEnd=True)
    _logger.error("u is\n {0}".format(u))
    return (mr, u)


# s is r*1, h is r*n and possibly non systematic, t is weight of the error (i.e. # of ones)
# return error vector e (n*1) s.t. He^ = s AND weight(e) = t
def isd(s, t, h):
    r = h.shape[0]
    n = h.shape[1]
    k = n - r
    k_zeros = np.zeros(k)
    _logger.debug("\nr={0}, n={1}, k={2}".format(r, n, k))
    _logger.debug("\ns={0}, t={1}, H=\n{2}".format(s, t, h))

    exit_condition = False
    # From now on exit_condition is used to continue the algorithm until we found
    # the right weight for the error
    while (not exit_condition):
        # p stands for permutation matrix
        hp, p = permute(h)
        # tr stands for transformation matrix
        hr, u = rref(hp)

        while (all(item is None for item in (hr, u))):
            _logger.debug("None returned, retrying")
            hp, p = permute(h)
            hr, u = rref(hp)
        hr = hr % 2
        u = u % 2
        p = p % 2
        _logger.debug("p is \n{0}".format(p))
        _logger.debug("h is \n{0}".format(hr))
        _logger.debug("u is \n{0}".format(u))

        tst = hr[:, k:n]
        id = np.eye(r)
        np.testing.assert_almost_equal(tst, id)

        # Apply Us to obtain s_signed
        s_sig = np.dot(u, s) % 2
        _logger.debug("s signed is {0}".format(s_sig))
        # e_hat is [0_{1*k} S_signed^transposed]
        e_hat = np.concatenate([k_zeros, s_sig.T])
        _logger.debug("e hat is {0}".format(e_hat))

        # TODO check weight of e_hat
        t_hat = np.sum(e_hat)
        _logger.debug(t_hat)
        exit_condition = t_hat == t

    e = np.dot(e_hat, p.T)
    _logger.debug("s was {0}, e is {1}".format(s, e))
    return e
