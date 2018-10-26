import hamming_code as hc
import rectangular_code as rc
import logging
import numpy as np


def test_743_codeword(u):

    g = hc.get_fixed_hamming_generator_743()
    c = rc.get_codeword_of_message(u, g)
    print("Original message is {0}".format(u))
    print("Generated codeword is {0}".format(c))
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
    print("Syndrome is {0}".format(s))
    return s


def main():
    global logger
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    u = np.array([1, 0, 0, 1])
    c = test_743_codeword(u)
    e = np.array([0, 1, 0, 0, 0, 0, 0])
    ctilde = (c + e) % 2
    s = test_743_syndrome(u, e, ctilde)


if __name__ == "__main__":
    main()
