import hamming_codes as hc
import rectangular_code as rc
import logging
import numpy as np


def test_743_codeword(u):

    g = hc.get_fixed_hamming_generator_743()
    c = rc.get_codeword_of_message(u, g)
    _logger.debug("Original message is {0}".format(u))
    _logger.debug("Generated codeword is {0}".format(c))
    # test_944(, np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]))
    return c


def test_743_syndrome(u, e, ctilde):
    """FIXME! briefly describe function

    :param u: original message vector
    :type u: 1xk matrix
    :param e: error vector
    :param e: 1xn error vector
    :param ctilde: erroneous codeword received
    :param ctilde: 1xn codeword vector
    :returns: null

    """
    h = hc.get_parity_matrix_systematic(7, 4)
    s = np.dot(h, e.T)
    _logger.debug("Syndrome is {0}".format(s))
    return s


def main():
    u = np.array([1, 0, 0, 1])
    c = test_743_codeword(u)
    e = np.array([0, 1, 0, 0, 0, 0, 0])
    ctilde = (c + e) % 2
    s = test_743_syndrome(u, e, ctilde)

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
_handler.setFormatter(_formatter)
if (_logger.hasHandlers()):
    _logger.handlers.clear()
_logger.addHandler(_handler)
_logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    main()
