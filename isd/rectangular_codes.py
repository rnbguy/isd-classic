import numpy as np
import sympy
import rectangular_code
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


def get_fixed_generator_944():
    """
    Get a fixed generator matrix G for the [9, 4, 4] code
    """
    i4 = np.eye(4)
    # i4 = np.eye(4, dtype=np.bool)
    a45 = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1],
                    [1, 1, 1, 1]]).T

    g = np.concatenate((i4, a45), axis=1)
    _logger.debug("***G***\n{0}".format(g))
    print("TESTING")
    h_rref, cols = sympy.Matrix(g).rref()
    print(h_rref)
    print(cols)
    print("/TESTING")
    return g
