"""A simple module to compute the # of identity matrices and # of correct
weights found using a classical algorithm identical to the quantum algorithm
implementation.
"""
from itertools import combinations
try:  # python >= 3.8
    from math import comb
except ImportError:
    from scipy.special import comb

import numpy as np
from experiments.quantemu.rref_reversible import rref
import operator

# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool


def _go_support(h, h_cols, isdstar_cols, iden):
    isd_cols = sorted(tuple(h_cols - set(isdstar_cols)))
    h_rref = h.copy()
    rref(h_rref, isdstar_cols, syn=None, u=None)
    h_right = h_rref[:, isdstar_cols]
    isiden = np.array_equal(h_right, iden)
    return isiden


def go(h, pool: Pool):
    r, n = h.shape
    iden = np.eye(r)
    tot_iden = 0
    h_cols = set(range(n))
    # the 2 definitions should be useless, but avoid later errors in case of no
    # iterations is equal to 0
    tot_iter = 0
    print("-" * 20)
    print(f"tot_iter: expected [binom(n,r)] = {comb(n,r)}")
    ress = []
    for tot_iter, isdstar_cols in enumerate(combinations(range(n), r)):
        res = pool.apply_async(_go_support, (h, h_cols, isdstar_cols, iden))
        ress.append(res)

    tot_iter += 1
    print(f"tot_iter: real = {tot_iter}")

    print("-" * 20)
    print(f"# identity matrices")
    # print(f"expected [.288 * tot_iter]: {.288*(2**(r*r))}")
    print(f"expected [.288 * tot_iter]: {.288*tot_iter}")
    tot_iden = sum(map(operator.methodcaller('get'), ress))
    print(f"real = {tot_iden}")

    # print("Some stats")
    print("-" * 20)
    print(f"% identities = tot_iden / tot_iter")
    print(f"expected: [prod_(i=1)(r)(1-1/2^i)] = .288")
    print(f"real: {tot_iden / tot_iter}")
    print("*" * 30)


def gen_random_matrix_and_rank_check(r, n, rank_check=False):
    rng = np.random.default_rng()
    # Discrete uniform distribution
    h = rng.integers(2, size=(r, n))
    rank = np.linalg.matrix_rank(h)
    if rank_check:
        while rank != r:
            h = rng.integers(2, size=(r, n))
            rank = np.linalg.matrix_rank(h)
    return h


def main():
    rank_check = True
    print(f"rank_check {rank_check}")
    h = gen_random_matrix_and_rank_check(10, 19, rank_check=rank_check)
    pool_size = 12
    pool = Pool(pool_size)
    go(h, pool=pool)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
