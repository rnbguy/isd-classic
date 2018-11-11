import numpy as np
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


# It uses a sort of LPU decomposition, where we're not really interested in
# L and P separately, but as a whole matrix applied to the matrix to obtain the
# RREF.
# I.e., given the original matrix M, we want to obtain its RREF form U and the
# corresponding L matrix s.t. L*M = U.
def get_rref(m, stopAt=3, dtype=np.float, startAtEnd=False):
    """FIXME! briefly describe function

    :param m: The matrix to be reduced
    :param stopAt: 1 if we want only the REF form, 2 if we want the RREF form, 3 (default) if we want the unitary RREF
    :param dtype: the dtype of the array (default np.float)
    :param startAtEnd: You want to leave it at False for most of the cases. It is set to True to have the systematic form of the parity matrix H, where the unitary RREF should be in the last positions (i.e. H = [ A | I_{square - square_start}], where I is the identity matrix, i.e. our unitary RREF)
    :returns: (P, L, U): L the transformation matrix s.t. L*M = U; U is the reduced matrix; P is the permutation matrix for the rows (useless atm). (None, None, None) if reduction is not possible.
    :rtype: tuple of np.array

    """
    # Multiplied by 1 bcz if dtype is bool, in this way we show a matrix of 
    # 0's and 1's instead of True and False
    _logger.debug("M = \n{0}".format(1 * m))
    # Make a copy of the original matrix, just in case it may serve.
    # Maybe it's useless, but since python passes references to objects by values
    # (i.e. like in Java) https://stackoverflow.com/a/986145/2326627
    u = m
    # This is used bcz the matrix is not surely a square matrix, so
    # we're going to consider the maximum b/w number of rows and columns as the
    # size of the square matrix.
    square = u.shape[0] if (u.shape[0] < u.shape[1]) else u.shape[1]

    _logger.debug("square = {0}".format(square))

    # If startAtEnd is set, the square matrix should be constructed in the right part of the matrix
    square_start = u.shape[1] - square if (startAtEnd) else 0
    _logger.debug("Square start is = {0}".format(square_start))

    # ltot will cointain all the transformations done on the original matrix
    # At the beginning, it's obv an identity matrix
    ltot = np.eye(square, dtype=dtype)
    # ptot should cointain all the row permutation done to the original matrix
    # It's really useless in our functions, but we track it just for future reference
    ptot = np.eye(square, dtype=dtype)

    if (stopAt < 1 or stopAt > 3):
        exit("Error stopAt")
    _logger.debug("*** PART 1: Row Echelon Form")
    ptot, ltot, u = _ref(u, square, square_start, ltot, ptot, dtype)
    _logger.debug("*** PART 1: Completed")
    if (stopAt == 1):
        return (ptot, ltot, u)
    if (all(item is None for item in (ptot, ltot, u))):
        _logger.debug("None tuple returned")
        return (None, None, None)

    _logger.debug("*** PART 2: Reduced Row Echelon Form *********")
    ptot, ltot, u = _rref(u, square, square_start, ltot, ptot, dtype)
    _logger.debug("*** PART 2: Completed")
    if (stopAt == 2):
        return (ptot, ltot, u)
    if ((ptot, ltot, u) is (None, None, None)):
        _logger.debug("None type returned")
        return (None, None, None)

    _logger.debug("*** PART 3: Normalizing coeffients on diagonal *********")
    ptot, ltot, u = _normalize(u, square, square_start, ltot, ptot, dtype)
    _logger.debug("*** PART 3: Completed")
    _logger.debug("***\nLTOT=\n{0}\nU=\n{1}".format(ltot, u))
    return (ptot, ltot, u)


# A REF is a triangular matrix with all 0's below the diagonal
def _ref(u, square, square_start, ltot, ptot, dtype):
    for i in range(square_start, square_start + square):  # columns, 0.. 3
        l = np.eye(square, dtype=dtype)
        p = np.eye(square, dtype=dtype)

        # Checking if there are 0's on the diagonal, if so try to swap the row
        # containing the zero with one of the rows below it
        exc_row = i + 1
        while (u[i - square_start][i] == 0):
            _logger.debug("0 at i = {0}".format(i))
            # revert back u and p at their original states at each iteration
            # First time it's useless
            u = np.dot(p.T, u)
            p = np.dot(p.T, p)

            if (exc_row >= square):
                # In this case, we've been unable to swap rows
                _logger.debug(
                    "Unable to swap rows, returning (None, None, None)")
                return (None, None, None)
            # swap rows
            # The first is a random solution
            # p = np.random.permutation(p)
            # u = np.dot(p, u)
            # However, we should only swap the ones below i
            p[[i, exc_row]] = p[[exc_row, i]]
            u = np.dot(p, u)
            exc_row += 1
            _logger.debug("Permuted U = \n{0}\n".format(1 * u))

        ptot = np.dot(p, ptot)
        ltot = np.dot(p, ltot)

        for j in range(i - square_start + 1, square):  #rows, i+1 .. 3
            _logger.debug("i = {0}, j = {1}".format(i, j))
            l[j][i - square_start] = -(u[j][i] / u[i - square_start][i])
        ltot = np.dot(l, ltot)
        u = np.dot(l, u)
        _logger.debug("\nL = \n{0}\nU =\n{1}".format(1 * l, 1 * u))

    return (ptot, ltot, u)


# A RREF is a triangular matrix with all 0's below and above the diagonal
# (i.e. all non-zero values are on the diagonal)
def _rref(u, square, square_start, ltot, ptot, dtype):
    for i in range(square_start + square - 1, square_start,
                   -1):  # columns, 2..1
        l = np.eye(square, dtype=dtype)
        p = np.eye(square, dtype=dtype)

        exc_row = i + 1
        while (u[i - square_start][i] == 0):
            _logger.debug("0 at i = {0}".format(i))
            # revert back u and p at their original states at each iteration
            # First time it's useless
            u = np.dot(p.T, u)
            p = np.dot(p.T, p)

            if (exc_row == square - 1):
                exit("Error, unable to swap rows")
            # swap rows
            # Random solution, but we should only swap the ones below i
            # p = np.random.permutation(p)
            # u = np.dot(p, u)
            p[[i, exc_row]] = p[[exc_row, i]]
            u = np.dot(p, u)
            exc_row += 1
            _logger.debug("Permuted U = \n{0}\n".format(1 * u))

        ptot = np.dot(p, ptot)
        ltot = np.dot(p, ltot)

        # i = 6, 5, 4
        # j = (1, 0), (0)
        # i = 2, 1, 0
        # j = (1, 0), (0)
        for j in range(i - square_start - 1, -1, -1):  # i-1..0
            _logger.debug("i = {0}, j = {1}".format(i, j))
            l[j][i - square_start] = -(u[j][i] / u[i - square_start][i])
        ltot = np.dot(l, ltot)
        u = np.dot(l, u)
        _logger.debug("\nL = \n{0}\nU =\n{1}".format(1 * l, 1 * u))

    return (ptot, ltot, u)


# We put all the non-zero entries of the diagonal to 1
def _normalize(u, square, square_start, ltot, ptot, dtype):
    l = np.eye(square, dtype=dtype)
    for i in range(square_start, square_start + square):
        _logger.debug("i = {0}".format(i))
        if (u[i - square_start][i] == 0):
            return (None, None, None)
        if (u[i - square_start][i] != 1):
            _logger.debug("{0} != 1".format(u[i - square_start][i]))
            l[i - square_start][i - square_start] = 1 / u[i - square_start][i]
            _logger.debug("l[i][i] = {0}".format(
                l[i - square_start][i - square_start]))
    u = np.dot(l, u)
    _logger.debug("\nL = \n{0}\nU =\n{1}".format(1 * l, 1 * u))
    ltot = np.dot(l, ltot)
    # _logger.debug("\nL = \n{0}\nU =\n{1}\nLTOT =\n{2}".format(l, u, ltot))

    return (ptot, ltot, u)
