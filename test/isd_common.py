import logging
import os
import unittest
import numpy as np
import inspect
import time
from datetime import datetime


class ISDTest(unittest.TestCase):
    MAX_N_SYNDROMES = 20

    def setUp(self):
        self._started_at = time.time()
        now = datetime.now().strftime("%H:%M:%S")
        print(now, end='->', flush=True)

    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('({:.2f}s)'.format(round(elapsed, 2)), end='', flush=True)

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(cls.__name__)
        cur_file = inspect.getfile(cls)
        cls.dirName = os.path.dirname(cur_file)
        cls.fileName = os.path.splitext(os.path.basename(cur_file))[0]
        cls.moduleName = os.path.splitext(cur_file)[0]
        if (os.getenv('LOG_LEVEL')):
            logging_level = logging._nameToLevel.get(
                os.getenv('LOG_LEVEL'), logging.info)
            # stream_handler = logging.StreamHandler()
            # stream_handler_formatter = logging.Formatter(
            #     '%(asctime)s %(levelname)-8s %(name)-12s %(funcName)-12s %(message)s'
            # )
            # stream_handler.setFormatter(stream_handler_formatter)
            log_file_name = os.path.join(cls.dirName, 'logs',
                                         '{}.log'.format(cls.fileName))
            log_fmt = ('%(asctime)s:%(module)s.%(funcName)s:%(levelname)s:'
                       ' %(message)s')
            formatter = logging.Formatter(log_fmt)
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(formatter)
            cls.logger.setLevel(logging_level)
            # cls.logger.addHandler(stream_handler)
            cls.logger.addHandler(file_handler)

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
