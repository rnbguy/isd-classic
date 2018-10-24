import numpy as np


def test_944(u, e, ctilde):
    """FIXME! briefly describe function

    :param u: original message vector
    :type u: 1xk matrix
    :param e: error vector
    :param e: 1xn error vector
    :param ctilde: erroneous codeword received
    :param ctilde: 1xn codeword vector
    :returns: null
    :rtype: 

    """
    n = 9
    k = 4
    r = n - k
    i4 = np.eye(k, dtype=np.bool)
    a45 = np.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]],
        dtype=np.bool).T

    g = np.concatenate((i4, a45), axis=1)
    print("***G***")
    print(g)

    c = np.dot(u, g) % 2
    print("***c***")
    print(c)

    a45t = a45.T
    ir = np.eye(r, dtype=np.bool)
    h = np.concatenate((a45t, ir), axis=1)
    print("***H***")
    print(h)

    s = np.dot(h, e.T)
    print("***s***")
    print(s)


test_944(np.array([1, 1, 1, 1]), np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]))
