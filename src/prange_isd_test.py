import numpy as np
import logging
import prange_isd
import os


def assert_15_11_4_w1():
    h = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                  [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]])
    syndromes = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0],
         [0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1],
         [1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]])
    errors = np.array(
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

    for i in range(5):
        # Trying to shuffle at random columns of h_systematic
        # (and also error_patterns_systematic accordingly) to have a better
        # test coverage of all the possible combinations of h
        p = np.random.permutation(np.eye(h.shape[1]))
        h_p = np.dot(h, p)
        errors_p = np.dot(errors, p)
        for i, s in enumerate(syndromes):
            _logger.debug("Launching prange with s = {0}".format(s))
            e = prange_isd.isd(s, 1, h_p)
            _logger.debug("For s = {0}, w = 1, h = \n{1}\nerror is {2}".format(
                s, h_p, e))
            _logger.debug("ASSERTING TEST RESULTS ...")
            np.testing.assert_array_almost_equal(e, errors_p[i])
            _logger.debug("... OK")


def assert_7_4_3_w1():
    h_systematic = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1],
                             [1, 0, 0], [0, 1, 0], [0, 0, 1]]).T

    syndromes = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                          [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    error_patterns_systematic = np.array([[0, 0, 0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 0, 0],
                                          [0, 1, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 0]])

    #for s in ([[1, 0, 1]]):
    for i in range(5):
        # Trying to shuffle at random columns of h_systematic
        # (and also error_patterns_systematic accordingly) to have a better
        # test coverage of all the possible combinations of h
        p = np.random.permutation(np.eye(h_systematic.shape[1]))
        h = np.dot(h_systematic, p)
        error_patterns = np.dot(error_patterns_systematic, p)
        for i, s in enumerate(syndromes):
            _logger.debug("Launching prange with s = {0}".format(s))
            e = prange_isd.isd(s, 1, h)
            _logger.debug("For s = {0}, w = 1, h = \n{1}\nerror is {2}".format(
                s, h, e))
            _logger.info("ASSERTING TEST RESULTS ...")
            np.testing.assert_array_almost_equal(e, error_patterns[i])
            _logger.info("... OK")


def main():
    if (os.getenv('LOG_LEVEL')):
        stream_handler = logging.StreamHandler()
        stream_handler_formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(name)-12s %(funcName)-12s %(message)s'
        )
        stream_handler.setFormatter(stream_handler_formatter)
        logging_level = logging._nameToLevel.get(
            os.getenv('LOG_LEVEL'), logging.info)
        _logger.setLevel(logging_level)
        _logger.addHandler(stream_handler)
        # Try to use the same log level also for prange_isd module
        prange_isd_logger = logging.getLogger('prange_isd')
        prange_isd_logger.setLevel(logging_level)
        prange_isd_logger.addHandler(stream_handler)
    assert_7_4_3_w1()
    assert_15_11_4_w1()


_logger = logging.getLogger(__name__)
if __name__ == "__main__":
    main()
