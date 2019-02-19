import numpy as np
import logging
from isdclassic.utils import lu
import itertools
#TODO delete
import sys

logger = logging.getLogger(__name__)


def get_matrix_before_bruteforce(h, s, t):
    logger.debug("*******")
    logger.debug("s={0}, t={1}, H=\n{2}".format(s, t, h))
    r = h.shape[0]
    n = h.shape[1]
    k = n - r
    logger.debug("r={0}, n={1}, k={2}".format(r, n, k))
    hr, u = None, None

    # From now on exit_condition_rref is used to continue the algorithm until we found
    # the right rref, i.e. having the identity matrix on the right
    exit_condition_rref = False

    # Trying to permute and then obtain the RREF
    while (not exit_condition_rref):
        # p stands for permutation matrix, hp is the permuted version of h
        # We are trying to permute the columns of H in such a way that the
        # columns of the information set I, with |I| = k, are packed to the
        # left of h.
        hp, perm = _permute(h)
        # hr stands for the matrix put in RREF, u for the transformation matrix
        # applied to obtain the RREF from the original matrix
        # We are trying to get the RREF of hp with the identity matrix r x r
        # placed to the right of hp
        hr, u = _rref(hp)
        # If rref returns None, it means that reduction was not possible,
        # i.e. the rightmost r x r matrix is not full-rank (different from
        # the id matrix in our case.
        exit_condition_rref = not (all(item is None for item in (hr, u)))
        if exit_condition_rref:
            logger.debug("EXIT CONDITION RREF IS TRUE, GOING TO CHECK WEIGHT")
        else:
            logger.debug("exit condition rref is false, retrying")

    logger.debug("perm is \n{0}".format(perm))
    logger.debug("hr, that is u.h.p is \n{0}".format(hr))
    logger.debug("u is \n{0}".format(u))

    # Apply U to s to obtain s_signed and applies mod2 to only obtain bits
    s_sig = np.mod(np.dot(u, s), 2)
    logger.debug("s signed is {0}".format(s_sig))
    return hr, u, perm, s_sig


def bruteforce(hr, t, p, s_sig):
    r = hr.shape[0]
    n = hr.shape[1]
    k = n - r
    # e_hat = np.concatenate((np.zeros(k), s_sig.T))
    # error = np.zeros(n)
    logger.debug("t is {}".format(t))
    logger.debug("k is {}".format(k))
    logger.debug("p is {}".format(p))
    # logger.debug("e hat is {}".format(e_hat))

    for i in itertools.combinations(range(k), p):
        logger.debug("i is {}".format(i))
        # extract only the columns indexed by i
        h_extr = hr[:, i]
        # sum the columns by rows
        sum_to_s = (h_extr.sum(axis=1) + s_sig) % 2
        logger.debug("sum to s is {}".format(sum_to_s))
        sum_to_s_w = np.sum(sum_to_s)
        logger.debug("sum to s w is {}".format(sum_to_s_w))
        e_hat = np.concatenate((np.zeros(k), sum_to_s))
        logger.debug("e hat is {}".format(e_hat))
        # return e_hat
        if sum_to_s_w == t - p:
            logger.debug("FOUND!! ")
            # sys.stdin.readline()
            for j in i:
                # a = [0] * (j - 1)
                # b = [1]
                # c = [0] * (n - len(a) - 1)
                # print(j)
                # print(a)
                # print(b)
                # print(c)
                # # e_hat += np.concatenate(([0] * (j - 1), [1], [0] * (n - j)))
                # e_hat += np.concatenate((a, b, c))
                # e_hat %= 2
                e_hat[j] = 1
            logger.debug("e_hat is {}".format(e_hat))
            # sys.stdin.readline()
            return e_hat


def run(h, s, t, p):
    """Run the isd algorithm

    :param s: the (n-k)x1 syndrome vector
    :param t: the weight of the error (i.e. the number of ones)
    :param h: the parity matrix, possibly in nonsystematic form
    :returns: the nx1 error vector s.t. He.T = s AND weight(e) = t
    :rtype: numpy.array

    """
    exit_condition = False
    while (not exit_condition):
        hr, u, perm, s_sig = get_matrix_before_bruteforce(h, s, t)
        e_hat = bruteforce(hr, t, p, s_sig)
        if np.sum(e_hat) == t:
            exit_condition = True
    e = np.mod(np.dot(e_hat, perm.T), 2)
    return e


# Random permutation of columns (by default) or rows
# Return the permutated matrix and the permutation matrix utilized
def _permute(m, cols=True):
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
def _rref(m):
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
    _, u, mr = lu.get_rref(m, startAtEnd=True, mod=2)
    #logger.debug("u is\n {0}".format(u))
    return (mr, u)
