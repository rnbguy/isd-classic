import rectangular_code as rc
import rectangular_codes as rcs
import logging
import numpy as np


def test_944_codeword(u):

    g = rcs.get_fixed_generator_944()
    c = rc.get_codeword_of_message(u, g)
    print("Original message is {0}".format(u))
    print("Generated codeword is {0}".format(c))
    # test_944(, np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]))


def test_944_syndrome(u, e, ctilde):
    """FIXME! Not working

    :param u: original message vector
    :type u: 1xk matrix
    :param e: error vector
    :param e: 1xn error vector
    :param ctilde: erroneous codeword received
    :param ctilde: 1xn codeword vector
    :returns: null

    """
    pass

    ir = np.eye(r)
    h = np.concatenate((a45t, ir), axis=1)
    logger.debug("***H***\n{0}".format(h))

    s = np.dot(h, e.T)
    logger.debug("***s***\n".format(s))


def main():
    u = np.array([1, 1, 1, 1])
    test_944_codeword(u)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    main()
