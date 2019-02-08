import numpy as np
import logging

_logger = logging.getLogger(__name__)


def get_codeword_of_message(u, g):
    """
    Given an original message and a generation matrix G, returns the corresponding codeword.

    :param u: the original vector message, 1xk
    :param g: the generation matrix kxn
    :returns: codeword vector 1xk
    :rtype: numpy.array
    """
    c = np.dot(u, g) % 2
    return c


def get_syndrome_of_received_message(y, h):
    """
    Given a 1 x n received message and an r x n parity check matrix,
    returns the syndrome of the received message (i.e. the syndrome of the
    error)

    :param y: the received message, 1 x n np.array
    :param h: the parity check matrix, r x n
    :returns: syndrome of the message, 1 x r
    :rtype: np.array

    """
    s = np.dot(h, y.T).T % 2
    return np
