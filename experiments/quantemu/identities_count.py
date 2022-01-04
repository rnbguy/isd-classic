"""A simple module to compute the # of identity matrices and # of correct
weights found using a classical algorithm identical to the quantum algorithm
implementation.
"""
from itertools import combinations
from math import comb

import numpy as np
from isdclassic.utils import rectangular_codes_generation as rcg
from isdclassic.utils import rectangular_codes_hardcoded as rch
