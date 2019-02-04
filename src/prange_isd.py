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
_logger.setLevel(logging.INFO)


# Random permutation of columns (by default) or rows
# Return the permutated matrix and the permutation matrix utilized
def permute(m, cols=True):
    """
    Random permutation of columns (by default) or rows.

    :param m: the original matrix as a numpy.array
    :param cols: True (default) to permute the columns; False to permute the rows.
    :returns: The permutated matrix mp and the permutation matrix used p
    :rtype: tuple(numpy.array, numpy.array)

    """
    # length of columns or rows
    axis = m.shape[1] if cols else m.shape[0]
    # Create an identity matrix I and permute its columns to obtain
    # a permutation matrix
    i = np.eye(axis)
    p = np.random.permutation(i)

    # Post-multiply to permute columns, pre-multiply to permute rows
    # No need to do modulo operations since p contains only ones.
    mp = np.dot(m, p) if cols else np.dot(p, m)
    return (mp, p)


#
# Returns:
# - MR, the matrix M in Reduced Row Echelon Form (and also unitary)
# - U, the matrix used to obtain the Reduced Row Echelon Form
def rref(m):
    """Uses LU decomposition w/ Doolittle algorithm, i.e. PA = LU.

    :param m: the original matrix as numpy.array
    :returns: mr, the matrix M in RREF (and also unitary); U the matrix used to obtain the RREF
    :rtype: tuple

    """
    # WARNING: The (ptot, ltot, u) returned from get_rref() are different from ours.
    # Basically, the returned u is the original matrix put in RREF (so u corresponds to our mr);
    # ltot is the matrix of transformations applied to the original matrix to
    # obtain the RREF (so ltot corresponds to our u)
    # Note that the 1st parameter returned by the get_rref function is not used
    _, u, mr = utils.get_rref(m, startAtEnd=True, mod=2)
    #_logger.debug("u is\n {0}".format(u))
    return (mr, u)


def isd(s, t, h):
    """Run the isd algorithm

    :param s: the (n-k)x1 syndrome vector
    :param t: the weight of the error (i.e. the number of ones)
    :param h: the parity matrix, possibly in nonsystematic form
    :returns: the nx1 error vector s.t. He.T = s AND weight(e) = t
    :rtype: numpy.array

    """
    _logger.debug("s={0}, t={1}, H=\n{2}".format(s, t, h))
    r = h.shape[0]
    n = h.shape[1]
    k = n - r
    _logger.debug("r={0}, n={1}, k={2}".format(r, n, k))

    # From now on exit_condition_weight is used to continue the algorithm until we found
    # the right weight for the error
    exit_condition_weight = False
    while (not exit_condition_weight):
        # From now on exit_condition_rref is used to continue the algorithm until we found
        # the right rref, i.e. having the identity matrix on the right
        exit_condition_rref = False

        # Trying to permute and then obtain the RREF
        while (not exit_condition_rref):
            # p stands for permutation matrix, hp is the permuted version of h
            # We are trying to permute the columns of H in such a way that the
            # columns of the information set I, with |I| = k, are packed to the
            # left of h.
            hp, p = permute(h)
            # hr stands for the matrix put in RREF, u for the transformation matrix
            # applied to obtain the RREF from the original matrix
            # We are trying to get the RREF of hp with the identity matrix r x r
            # placed to the right of hp
            hr, u = rref(hp)
            # If rref returns None, it means that reduction was not possible,
            # i.e. the rightmost r x r matrix is not full-rank (different from
            # the id matrix in our case.
            exit_condition_rref = not(all(item is None for item in (hr, u)))
            if exit_condition_rref:
                _logger.debug("EXIT CONDITION RREF IS TRUE, GOING TO CHECK WEIGHT")
            else:
                _logger.debug("exit condition rref is false, retrying")

        _logger.debug("p is \n{0}".format(p))
        _logger.debug("hr, that is u.h.p is \n{0}".format(hr))
        _logger.debug("u is \n{0}".format(u))

        # Double check that the right hand side, (n-k)*(n-k)=r*r matrix is an
        # identity matrix (and so H is in standard form)
        # Commented out to improve speed
        # tst = hr[:, k:n]
        # id = np.eye(r)
        # np.testing.assert_almost_equal(tst, id)

        # Apply U to s to obtain s_signed and applies mod2 to only obtain bits
        s_sig = np.mod(np.dot(u, s), 2)
        _logger.debug("s signed is {0}".format(s_sig))
        # check weight of s_sig; if it's equal to t, we exit the loop
        # bcz we've found the correct e_hat.
        # In reality, we should check for the weight of e_hat, but the latter 
        # is the concatenation of zeros and s_sig, so we anticipate the test
        t_hat = np.sum(s_sig)
        _logger.debug("Weight of s is {0}".format(t_hat))
        exit_condition_weight = t_hat == t
        if exit_condition_weight:
            _logger.debug("WEIGHT IS CORRECT, FOUND e")
            # e_hat is the concatenation of all zeros 1xk vector and s_signed^transposed
            e_hat = np.concatenate([np.zeros(k), s_sig.T])
            _logger.info("s signed is {0}".format(s_sig))
            _logger.info("e hat is {0}".format(e_hat))
            _logger.info("p is \n{0}".format(p))
            _logger.info("u is \n{0}".format(u))
            _logger.info("hr, that is u.h.p is \n{0}".format(hr))
        else:
            _logger.debug("Weight is wrong, retrying")

    # return the error vector multiplying e_hat by the permutation matrix
    e = np.mod(np.dot(e_hat, p.T), 2)
    _logger.info("s was {0}, e is {1}".format(s, e))
    return e
