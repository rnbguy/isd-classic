import numpy as np
import logging
from isd.methods import prange
from isd.utils.rectangular_codes_generation import generate_parity_matrix_nonsystematic_for_hamming

logger = logging.getLogger(__name__)


def ex_7_4():
    h = generate_parity_matrix_nonsystematic_for_hamming(7, 4)
    s = np.array([1, 1, 1])
    e = prange.isd(h, s, 1)
    print("H = \n{0}".format(h))
    print("For s = {0}, e = {1}".format(s, e))


def ex_15_11():
    h = generate_parity_matrix_nonsystematic_for_hamming(15, 11)
    s = np.array([0, 1, 1, 0])
    e = prange.isd(h, s, 1)
    print("H = \n{0}".format(h))
    print("For s = {0}, e = {1}".format(s, e))


def main():
    ex_7_4()
    ex_15_11()


if __name__ == "__main__":
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    main()
