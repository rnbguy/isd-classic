import numpy as np
import logging
import prange_isd
import hamming_code

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
_handler.setFormatter(_formatter)
if (_logger.hasHandlers()):
    _logger.handlers.clear()
_logger.addHandler(_handler)
_logger.setLevel(logging.DEBUG)


#Syndrome #Error Pattern # Works
#0 0 0    #0 0 0 0 0 0 0 # inf
#0 0 1    #0 0 0 0 0 0 1 # T
#0 1 0    #0 0 0 0 0 1 0 # T
#0 1 1    #0 0 0 1 0 0 0 # inf
#1 0 0    #0 0 0 0 1 0 0 # F
#1 0 1    #0 1 0 0 0 0 0 # inf
#1 1 0    #0 0 1 0 0 0 0 # inf
#1 1 1    #1 0 0 0 0 0 0 # inf
def test1():
    h = hamming_code.get_parity_matrix_nonsystematic(7, 4)
    s = np.array([1, 1, 1])
    return prange_isd.isd(s, 1, h)


def test2():
    h = hamming_code.get_parity_matrix_nonsystematic(15, 11)
    s = np.array([0, 1, 1, 0])
    return prange_isd.isd(s, 1, h)


def main():
    test1()
    # test2()


if __name__ == "__main__":
    main()
