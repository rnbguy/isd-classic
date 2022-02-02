#!/usr/bin/python3
"""A simple module to compute the # of identity matrices and # of correct
weights found using a classical algorithm identical to the quantum algorithm
implementation.
"""
import operator
import argparse
from itertools import combinations, product
import os

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

ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "OPENBLAS_MAIN_FREE",
               "NUMEXPR_NUM_THREADS")


def parse_arguments():
    parser = argparse.ArgumentParser(
        "Counts the zeros of the RREF reversible function")
    parser.add_argument("-n", type=int)
    parser.add_argument("-r", type=int)
    parser.add_argument("-k", type=int)
    parser.add_argument("-d", type=int)
    parser.add_argument("-t", type=int)
    parser.add_argument("-minp", type=int, help="starting p", default=0)
    parser.add_argument(
        "-maxp",
        type=int,
        help=
        ("ending p, by default it will be equal to t and it cannot be bigger than it"
         ))
    parser.add_argument("-j",
                        type=int,
                        help="# of parallel processes",
                        default=1)
    parser.add_argument("h_generator",
                        choices=('random', 'hamming', 'other'),
                        default='random',
                        nargs='?')
    parser.add_argument("--set_threads", action='store_true')
    namespace = parser.parse_args()

    if namespace.minp < 0:
        namespace.minp = 0
    if namespace.maxp is not None and namespace.maxp <= namespace.minp:
        raise argparse.ArgumentTypeError(f"maxp should be greater than minp")

    if namespace.h_generator == 'hamming':
        if not (namespace.r is not None and namespace.r > 0):
            raise argparse.ArgumentTypeError(
                f"For choice Hamming r must be > 0, provided {namespace.r}")
    elif namespace.h_generator == 'random':
        if (namespace.r is None or namespace.r < 1 or namespace.n is None
                or namespace.r < 1 or namespace.n <= namespace.r):
            raise argparse.ArgumentTypeError(
                f"For choice random n and r must be defined, with n>r, provided n: {namespace.n} and r: {namespace.r}"
            )
    elif namespace.h_generator == 'other':
        if (namespace.n is None or namespace.n < 1 or namespace.k is None
                or namespace.k < 1 or namespace.d is None or namespace.d < 3
                or namespace.t is None or namespace.t < 1):
            raise argparse.ArgumentTypeError(
                f"For choice other you must specify all parameters")
    return namespace


def _rref(h, isdstar_cols, syn, iden):
    _assert_environment()
    h_rref = h.copy()
    syn_sig = syn.copy() if syn is not None else None
    # U is used just for double check
    # u = np.eye(r, dtype=np.ubyte) if double_check else None
    rref(h_rref, isdstar_cols, syn=syn_sig, u=None)
    h_right = h_rref[:, isdstar_cols]
    isiden = np.array_equal(h_right, iden)
    return (h_rref, syn_sig, isdstar_cols, isiden)


def _weight(h_rref, isiden, syn_sig, v_cols, t, p):
    _assert_environment()
    # We proceed to extract independently from the identity matrix check,
    # simulating exactly the quantum circuit behaviour
    # TODO check sum with %2
    sum_to_s = (h_rref[:, v_cols].sum(axis=1) + syn_sig) % 2
    sum_to_s_w = np.sum(sum_to_s)
    is_correct_w = sum_to_s_w == t - p
    return (isiden, is_correct_w)


def go(h, t, syn, pool, minp, maxp, skip_count_identities=False):
    r, n = h.shape
    k = n - r
    # assert t - p >= 0
    iden = np.eye(r)
    h_cols = set(range(n))

    print(f"n {n} k {k} r {r}\nt {t}")
    tot_iter1 = comb(n, r)
    print(f"tot_iter1: expected [binom(n={n},r={r})] = {tot_iter1}")
    rref_ress = []

    for isdstar_cols in combinations(range(n), r):
        res = pool.apply_async(_rref, (h, isdstar_cols, syn, iden))
        rref_ress.append(res)
    print('rref done')
    # At this point we have all the results for all possible RREF
    n_idens = sum(
        i for _, _, _, i in map(operator.methodcaller('get'), rref_ress))
    print("-" * 20)
    print(f"# identity matrices")
    # print(f"expected [.288 * tot_iter]: {.288*(2**(r*r))}")
    print(f"expected [.288 * tot_iter1]: {.288*tot_iter1}")
    print(f"real = {n_idens}")

    # print("Some stats")
    print("-" * 20)
    print(f"% identities")
    print(f"expected: [prod_(i=1)(r)(1-1/2^i)] = .288")
    print(f"real: {n_idens / tot_iter1}")

    for p in range(minp, maxp + 1):
        print("-" * 20)
        tot_iter2 = comb(k, p)
        print(f"p = {p}")
        print(f"tot_iter2: expected [binom(k={k},p={p})]= {tot_iter2}")
        # print("*" * 30)

        weig_ress = []
        for (h_rref, syn_sig, isdstar_cols,
             isiden) in map(operator.methodcaller('get'), rref_ress):
            isd_cols = sorted(tuple(h_cols - set(isdstar_cols)))
            for v_cols in combinations(isd_cols, p):
                res = pool.apply_async(_weight,
                                       (h_rref, isiden, syn_sig, v_cols, t, p))
                weig_ress.append(res)
            # print('weight done')

        n_weights = 0
        n_weights_given_iden = 0

        for isiden, is_correct_w in map(operator.methodcaller('get'),
                                        weig_ress):
            if is_correct_w:
                n_weights += 1
                if isiden:
                    n_weights_given_iden += 1

        print("-" * 20)
        print(f"# Correct weights")
        # TODO this is only valid for Prange (i.e., p=0)
        print(f"expected ?= [binom(n-t={n-t},k={k})] {comb(n-t,k)}")
        print(f"real (independently of identity matrix) = {n_weights}")
        print(f"real (given matrix was identity) = {n_weights_given_iden}")
        print("-" * 20)
        print(f"% Correct weights")
        print(
            f"% total correct weights = tot_correct_weight / (tot_iter1 * tot_iter2): {n_weights / (tot_iter1* tot_iter2)}"
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


def _random(r: int, n: int):
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
    h, g, syndromes, errors, w, isHamming = rch.get_isd_systematic_parameters(
        n, k, d, w)
    return h, w, syndromes


def _prepare_environment():
    """This is necessary since numpy already uses a lot of threads. See
https://stackoverflow.com/a/58195413/2326627"""
    print("Setting threads to 1")
    for env in ENVIRONMENT:
        os.environ[env] = "1"

def _print_environment():
    for env in ENVIRONMENT:
        print(f"{env} = {os.environ[env]}")

def _assert_environment():
    for env in ENVIRONMENT:
        assert os.environ[env] == "1", _print_environment()



def main():
    print("#" * 70)
    namespace = parse_arguments()
    print(namespace)
    if namespace.j > 1 and namespace.set_threads:
        _prepare_environment()
    pool = Pool(namespace.j)
    if namespace.h_generator == 'random':
        h, t, syns = _random(namespace.r, namespace.n)
    elif namespace.h_generator == 'hamming':
        h, t, syns = _hamming_non_systematic(namespace.r)
    elif namespace.h_generator == 'other':
        h, t, syns = _other(namespace.n, namespace.k, namespace.d, namespace.t)
    else:
        raise Exception("Error in h generator")

    if namespace.maxp is None or namespace.maxp > namespace.t:
        namespace.maxp = t
    print(namespace)

    # iden_and_w(h, t, syns, pool, namespace.minp, namespace.maxp)
    go(h, t, syns[0], pool, namespace.minp, namespace.maxp)
    pool.close()
    pool.join()
    print("#" * 70)


if __name__ == '__main__':
    main()
