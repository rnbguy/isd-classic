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


# Stop at 3 is to stop the algorithm at any step before completion
# In reality we're not really interested, but to the overall transformations
# applied to a
def lu(a, stopAt=3, dtype=np.float):
    # u at the end will cointain the matrix in RREF
    u = a
    # This is used bcz the matrix is not surely a square matrix, so
    # we're going to consider the first square submatrix of the original one.
    square = u.shape[0] if (u.shape[0] < u.shape[1]) else u.shape[1]

    # ltot will cointain all the transformations done on the original matrix
    ltot = np.eye(square, dtype=dtype)
    # ptot should cointain all the row permutation done to the original matrix
    # It's really useless in our functions, but we track it just for future reference
    ptot = np.eye(square, dtype=dtype)

    if (stopAt < 1 or stopAt > 3):
        exit("Error stopAt")
    _logger.debug("*** PART 1: Row Echelon Form")
    ptot, ltot, u = ref(u, square, ltot, ptot, dtype)
    _logger.debug("*** PART 1: Completed")
    _logger.debug("Outcome:\nLTOT=\n{0}\nU=\n{1}".format(ltot, u))
    if (stopAt == 1):
        return (ptot, ltot, u)
    if (all(item is None for item in (ptot, ltot, u))):
        _logger.debug("None tuple returned")
        return (None, None, None)

    _logger.debug("*** PART 2: Reduced Row Echelon Form *********")
    ptot, ltot, u = rref(u, square, ltot, ptot, dtype)
    _logger.debug("*** PART 2: Completed")
    if (stopAt == 2):
        return (ptot, ltot, u)
    if ((ptot, ltot, u) is (None, None, None)):
        _logger.debug("None type returned")
        return (None, None, None)

    _logger.debug("*** PART 3: Normalizing coeffients on diagonal *********")
    ptot, ltot, u = normalize(u, square, ltot, ptot, dtype)
    _logger.debug("*** PART 3: Completed")
    return (ptot, ltot, u)


def ref(u, square, ltot, ptot, dtype):
    for i in range(square):  # columns, 0.. 3
        l = np.eye(square, dtype=dtype)
        p = np.eye(square, dtype=dtype)

        exc_row = i + 1
        while (u[i][i] == 0):
            _logger.debug("0 at i = {0}".format(i))
            # revert back u and p at their original states at each iteration
            # First time it's useless
            u = np.dot(p.T, u) % 2
            p = np.dot(p.T, p) % 2

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
            u = np.dot(p, u) % 2
            exc_row += 1
            _logger.debug("Permuted U = \n{0}\n".format(u))

        ptot = np.dot(p, ptot) % 2
        ltot = np.dot(p, ltot) % 2

        for j in range(i + 1, square):  #i+1 .. 3
            _logger.debug("i = {0}, j = {1}".format(i, j))
            l[j][i] = -(u[j][i] / u[i][i])
        ltot = np.dot(l, ltot) % 2
        u = np.dot(l, u) % 2
        _logger.debug("\nL = \n{0}\nU =\n{1}".format(l, u))

    return (ptot, ltot, u)


def rref(u, square, ltot, ptot, dtype):
    for i in range(square - 1, 0, -1):  # columns, 2..1
        l = np.eye(square, dtype=dtype)
        p = np.eye(square, dtype=dtype)

        exc_row = i + 1
        while (u[i][i] == 0):
            _logger.debug("0 at i = {0}".format(i))
            # revert back u and p at their original states at each iteration
            # First time it's useless
            u = np.dot(p.T, u) % 2
            p = np.dot(p.T, p) % 2

            if (exc_row == square - 1):
                exit("Error, unable to swap rows")
            # swap rows
            # Random solution, but we should only swap the ones below i
            # p = np.random.permutation(p)
            # u = np.dot(p, u)
            p[[i, exc_row]] = p[[exc_row, i]]
            u = np.dot(p, u) % 2
            exc_row += 1
            _logger.debug("Permuted U = \n{0}\n".format(u))

        ptot = np.dot(p, ptot) % 2
        ltot = np.dot(p, ltot) % 2

        for j in range(i - 1, -1, -1):  # i-1..0
            _logger.debug("i = {0}, j = {1}".format(i, j))
            l[j][i] = -(u[j][i] / u[i][i])
        ltot = np.dot(l, ltot) % 2
        u = np.dot(l, u) % 2
        _logger.debug("\nL = \n{0}\nU =\n{1}".format(l, u))

    return (ptot, ltot, u)


def normalize(u, square, ltot, ptot, dtype):
    l = np.eye(square)
    for i in range(square):
        _logger.debug("i = {0}".format(i))
        if (u[i][i] == 0):
            return (None, None, None)
        if (u[i][i] != 1):
            _logger.debug("{0} != 1".format(u[i][i]))
            l[i][i] = 1 / u[i][i]
            _logger.debug("l[i][i] = {0}".format(l[i][i]))
    ltot = np.dot(l, ltot) % 2
    u = np.dot(l, u) % 2
    _logger.debug("\nL = \n{0}\nU =\n{1}\nLTOT =\n{2}".format(l, u, ltot))

    return (ptot, ltot, u)


def test_bits():
    n = 15
    k = 11
    r = n - k
    support = np.array([i for i in range(1, n + 1)])
    h = (((support[:, None] & (1 << np.arange(r)))) > 0).T.astype(np.bool)

    p, l, u = lu(h, 3, np.bool)
    while (all(item is None for item in (p, l, u))):
        _logger.debug("*********** PERMUTING H *********************")
        h = np.random.permutation(h.T).T
        p, l, u = lu(h, 3, np.bool)

    # p = np.around(p, 4)
    # l = np.around(l, 4)
    # u = np.around(u, 4)
    print("H=\n{0}\nH_RREF=\n{1}\nLTOT=\n{2}\nPTOT=\n{3}".format(h, u, l, p))

    tst = np.dot(l, h) % 2
    print("Testing L * H, should be equal to A_RREF\n{0}".format(tst))
    # tst = np.dot(l.T, u) %2
    # print("Testing L.T * U, should be equal to A\n{0}".format(tst))
    # tst = np.dot(p, a)
    # print("Testing P * A, should be equal to A\n{0}".format(tst))


def test_zero():
    a = np.array([[0, 1, 1], [3, 1, 1], [8, 7, 9]], dtype=np.float)
    p, l, u = lu(a, 3, np.float)
    # p = np.around(p, 4)
    # l = np.around(l, 4)
    # u = np.around(u, 4)
    print("A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nPTOT=\n{3}".format(a, u, l, p))

    tst = np.dot(l, a) % 2
    print("Testing L * A, should be equal to A_RREF\n{0}".format(tst))
    # tst = np.dot(l.T, u)
    # print("Testing L.T * U, should be equal to A\n{0}".format(tst))
    # tst = np.dot(p, a)
    # print("Testing P * A, should be equal to A\n{0}".format(tst))


def test_pdf():
    a = np.array([[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]],
                 dtype=np.float)
    p, l, u = lu(a, 1)
    print("A=\n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\nPTOT=\n{3}".format(a, u, l, p))

    tst = np.dot(l, a) % 2
    print("Testing L * A, should be equal to A_RREF\n{0}".format(tst))
    tst = np.dot(l.T, u) % 2
    print("Testing L.T * U, should be equal to A\n{0}".format(tst))
    tst = np.dot(p, a) % 2
    print("Testing P * A, should be equal to A\n{0}".format(tst))


def test_wiki():
    a = np.array([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]])
    ar, lt = lu(a)
    print("A = \n{0}\nA_RREF=\n{1}\nLTOT=\n{2}\n".format(a, ar, lt))

    tst = np.dot(lt, a) % 2
    print("Testing LT * A, should be equal to A_RREF\n{0}".format(
        np.dot(lt, a))) % 2
    assert np.array_equal(ar, tst)


def test_sympy_rref(h):
    import sympy
    m = sympy.Matrix(h).rref()
    print(m)
