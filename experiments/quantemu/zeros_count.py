#!/usr/bin/python3
"""A simple module to compute the # of identity matrices and # of correct
weights found using a classical algorithm identical to the quantum algorithm
implementation.
"""
import operator
import argparse
from itertools import combinations

try:  # python >= 3.8
    from math import comb
except ImportError:
    from scipy.special import comb

# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool

import numpy as np
from experiments.quantemu.rref_reversible import rref
from isdclassic.utils import rectangular_codes_generation as rcg
from isdclassic.utils import rectangular_codes_hardcoded as rch


def parse_arguments():
    parser = argparse.ArgumentParser(
        "Counts the zeros of the RREF reversible function")
    parser.add_argument("-n", type=int)
    parser.add_argument("-r", type=int)
    parser.add_argument("-k", type=int)
    parser.add_argument("-d", type=int)
    parser.add_argument("-t", type=int)
    parser.add_argument("-j",
                        type=int,
                        help="# of parallel processes",
                        default=1)
    parser.add_argument("h_generator",
                        choices=('random', 'hamming', 'other'),
                        default='random',
                        nargs='?')
    namespace = parser.parse_args()
    if namespace.h_generator == 'hamming':
        assert (namespace.r is not None
                and namespace.r > 0), "For choice Hamming r must be > 0"
    elif namespace.h_generator == 'random':
        assert (namespace.r is not None and namespace.r > 0
                and namespace.n is not None and namespace.n > 0
                and namespace.n > namespace.r
                ), "For choice random n and r must be defined, with n>r"
    elif namespace.h_generator == 'other':
        assert (namespace.n is not None and namespace.n > 0
                and namespace.k is not None and namespace.k > 0
                and namespace.d is not None and namespace.d > 2
                and namespace.t is not None and namespace.t > 0
                ), "For choice other you must specify all parameters"

    return namespace


def _check_iden(h, isdstar_cols, v_cols, t, p, syn, iden):
    h_rref = h.copy()
    syn_sig = syn.copy() if syn is not None else None
    # U is used just for double check
    # u = np.eye(r, dtype=np.ubyte) if double_check else None
    rref(h_rref, isdstar_cols, syn=syn, u=None)
    h_right = h_rref[:, isdstar_cols]
    isiden = np.array_equal(h_right, iden)
    # We proceed to extract independently from the identity matrix check,
    # simulating exactly the quantum circuit behaviour
    sum_to_s = (h_rref[:, v_cols].sum(axis=1) + syn_sig) % 2
    sum_to_s_w = np.sum(sum_to_s)
    is_correct_w = sum_to_s_w == t - p
    return (isiden, is_correct_w)


def go(h, t, p, syn, pool):
    assert t - p >= 0
    r, n = h.shape
    k = n - r
    iden = np.eye(r)
    h_cols = set(range(n))
    # the 2 definitions should be useless, but avoid later errors in case of no
    # iterations is equal to 0
    tot_iter1 = comb(n, r)
    tot_iter2 = comb(k, p)
    print("-" * 20)
    print(f"tot_iter1: expected [binom(n={n},r={r})] = {tot_iter1}")
    print(f"tot_iter2: expected [binom(k={k},p={p})]= {tot_iter2}")
    ress = []

    for isdstar_cols in combinations(range(n), r):
        isd_cols = sorted(tuple(h_cols - set(isdstar_cols)))
        for v_cols in combinations(isd_cols, p):
            res = pool.apply_async(_check_iden,
                                   (h, isdstar_cols, v_cols, t, p, syn, iden))
            ress.append(res)

    # B
    n_idens = sum(
        i for i, _ in map(operator.methodcaller('get'), ress)) / tot_iter2
    n_weights = sum(j for _, j in map(operator.methodcaller('get'), ress))
    n_weights_given_iden = sum(
        j for i, j in map(operator.methodcaller('get'), ress) if i and j)
    # print(tot_iter1)
    # print(tot_iter2)
    # print(n_idens)
    # print(n_weights)
    # print(n_weights_given_iden)

    print("-" * 20)
    print(f"# identity matrices")
    # print(f"expected [.288 * tot_iter]: {.288*(2**(r*r))}")
    print(f"expected [.288 * tot_iter]: {.288*tot_iter1}")
    print(f"real = {n_idens}")

    # print("Some stats")
    print("-" * 20)
    print(f"% identities")
    print(f"expected: [prod_(i=1)(r)(1-1/2^i)] = .288")
    print(f"real: {n_idens / tot_iter1}")

    print("-" * 20)
    print(f"# Correct weights")
    # TODO this is only valid for Prange (i.e., p=0)
    print(f"expected = [binom(n-t={n-t},k={k})] {comb(n-t,k)}")
    print(f"real (independently of identity matrix) = {n_weights}")
    print(f"real (given matrix was identity) = {n_weights_given_iden}")
    print("-" * 20)
    print(f"% Correct weights")
    print(
        f"% total correct weights = tot_correct_weight / (tot_iter * tot_iter2): {n_weights / (tot_iter1* tot_iter2)}"
    )
    print(
        f"% total correct weights identity = tot_correct_weight_iden / (tot_iter1 * tot_iter2): {n_weights_given_iden / (tot_iter1 * tot_iter2)}"
    )
    print("*" * 30)


def _gen_random_matrix_and_rank_check(r, n):
    rng = np.random.default_rng()
    # Discrete uniform distribution
    h = rng.integers(2, size=(r, n))
    rank = np.linalg.matrix_rank(h)
    while rank != r:
        h = rng.integers(2, size=(r, n))
        rank = np.linalg.matrix_rank(h)
    return h


def iden_and_w(h, w, syndromes, pool):
    """Check both # of identities and # of correct syndrome weights. Basically, we
    also check that, after RREF, the syndrome has the correct weight.

    """
    r, n = h.shape
    k = n - r
    # for p in reversed(range(w + 1)):
    # for p in range(1, 3):
    for p in range(w + 1):
        print(f"n {n} k {k} r {r}\nt {w} p {p}")
        go(h, w, p, syndromes[0], pool)


def _random(r: int, n: int):
    r, n = 4, 12
    k = n - r
    h = _gen_random_matrix_and_rank_check(r, n)
    # For the Gilbert-Varshamov bound, the probability that, starting from a
    # random matrix, there exists a codeword with weight less than d is low. In
    # the hypothesis, d = \delta * n, and \delta < 1/2. We take 1/4 in our
    # case. To have d=5, we should have an n>=20
    d = (np.floor(.25 * n))
    assert d >= 3, f"d should be at least 3 to be able to do something {d}"
    assert k <= n - d + 1, "Not respecting Singleton bound"
    w = int(np.floor((d - 1) / 2))
    # Create a random error with weight t
    error = np.concatenate((np.ones(w), np.zeros(n - w)))
    np.random.shuffle(error)
    syndromes = [None]
    syndromes[0] = (h @ error).astype(np.uint8)
    return h, w, syndromes


def _hamming_non_systematic(r: int):
    n = 2**r - 1
    h = rcg.generate_parity_matrix_nonsystematic_for_hamming_from_r(r)
    assert h.shape == (r, n), "Matrix not corresponding"
    w = 1
    # generate an error with w=1
    error = np.zeros(n, dtype='uint16')
    error[n - 1] = 1
    np.random.shuffle(error)
    syndromes = []
    syndromes.append(((h @ error) % 2).astype(np.uint8))
    return h, w, syndromes


def _other(n: int, k: int, d: int, w: int):
    # n, k, d, w = 7, 4, 3, 1
    # n, k, d, w = 16, 11, 4, 1
    # n, k, d, w = 23, 12, 7, 3
    h, g, syndromes, errors, w, isHamming = rch.get_isd_systematic_parameters(n, k, d, w)
    return h, w, syndromes


def main():
    namespace = parse_arguments()
    print(namespace)
    # pool_size = 12
    pool = Pool(namespace.j)
    if namespace.h_generator == 'random':
       h, t, syns = _random(namespace.r, namespace.n)
    elif namespace.h_generator == 'hamming':
        h, t, syns = _hamming_non_systematic(namespace.r)
    elif namespace.h_generator == 'other':
        h, t, syns = _other(namespace.n, namespace.k, namespace.d, namespace.t)
    else:
        raise Exception("Error in h generator")

    iden_and_w(h, t, syns, pool)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
