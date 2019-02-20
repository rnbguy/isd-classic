import numpy as np
import logging
from isdclassic.methods.lee_brickell import LeeBrickell
from isdclassic.utils import rectangular_codes_hardcoded as rch

logger = logging.getLogger(__name__)


def ex_8_4():
    h, g, syndromes, errors, w, isHamming = rch._get_8_4_4_w2()
    s = syndromes[0]
    exp_e = errors[0]
    print("Syndrome is {}".format(s))
    lee = LeeBrickell(h, s, w, 1)
    e = lee.run()
    print("H = \n{0}".format(h))
    print("For s = {0}, e = {1}".format(s, e))
    print("Expected e {}".format(exp_e))
    print(lee.result)


def main():
    ex_8_4()


if __name__ == "__main__":
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    other_loggers = logging.getLogger('isdclassic.methods')
    other_loggers.setLevel(logging.DEBUG)
    other_loggers.addHandler(handler)
    main()
