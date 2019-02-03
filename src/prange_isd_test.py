import numpy as np
import logging
import prange_isd


def assert_743_fixed_h():
    h = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0],
                  [0, 1, 0], [0, 0, 1]]).T

    syndromes = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                          [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    error_patterns = np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0]])

    #for s in ([[1, 0, 1]]):
    for i, s in enumerate(syndromes):
        _logger.debug("Launching prange with s = {0}".format(s))
        e = prange_isd.isd(s, 1, h)
        _logger.debug("For s = {0}, w = 1, h = \n{1}\nerror is {2}".format(s, h, e))
        _logger.debug("ASSERTING TEST RESULTS ...")
        np.testing.assert_array_almost_equal(e, error_patterns[i])
        _logger.debug("... OK")


def main():
    assert_743_fixed_h()


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
