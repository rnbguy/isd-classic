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


def get_codeword_of_message(u, g):
    """FIXME! briefly describe function

    :param u: the original vector message, 1xk
    :param g: the generation matrix kxn
    :returns: codeword vector 1xk
    :rtype: numpy.array

    """
    n = g.shape[1]  # columns
    k = g.shape[0]  # rows
    r = n - k
    _logger.debug("n = {0}, k = {1},  r = {2}".format(n, k, r))

    c = np.dot(u, g) % 2
    _logger.debug("***c***\n{0}".format(c))
    return c
