import logging
import os
import unittest
import numpy as np


class ISDTest(unittest.TestCase):
    MAX_N_SYNDROMES = 20

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(cls.__name__)
        if (os.getenv('LOG_LEVEL')):
            stream_handler = logging.StreamHandler()
            stream_handler_formatter = logging.Formatter(
                '%(asctime)s %(levelname)-8s %(name)-12s %(funcName)-12s %(message)s'
            )
            stream_handler.setFormatter(stream_handler_formatter)
            logging_level = logging._nameToLevel.get(
                os.getenv('LOG_LEVEL'), logging.info)
            cls.logger.setLevel(logging_level)
            cls.logger.addHandler(stream_handler)

    @classmethod
    def get_max_syndromes_errors(cls, syndromes, errors):
        if syndromes.ndim > 1 and syndromes.shape[0] > cls.MAX_N_SYNDROMES:
            cls.logger.info("Too many syndromes, shuffling and slicing")
            perm2 = np.random.permutation(np.eye(syndromes.shape[0]))
            syndromes = np.dot(perm2, syndromes)[:cls.MAX_N_SYNDROMES, :]
            errors = np.dot(perm2, errors)[:cls.MAX_N_SYNDROMES, :]
        return syndromes, errors

    @classmethod
    def scramble_h_errors(cls, h, errors):
        perm = np.random.permutation(np.eye(h.shape[1]))
        h_p = np.dot(h, perm)
        errors_p = np.dot(errors, perm)
        return h_p, errors_p
