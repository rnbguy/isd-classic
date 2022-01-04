"""A simple module to compute the # of identity matrices and # of correct
weights found using a classical algorithm identical to the quantum algorithm
implementation.
"""
from itertools import combinations
from math import comb

import numpy as np
from experiments.quantemu.rref_reversible import rref
from isdclassic.utils import rectangular_codes_generation as rcg
from isdclassic.utils import rectangular_codes_hardcoded as rch


def go(h, t, p, syn, double_check=False, check_inner=True):
    if check_inner:
        assert t - p >= 0
    r, n = h.shape
    k = n - r
    iden = np.eye(r)
    tot_iden = 0
    tot_correct_weight = 0
    tot_correct_weight_iden = 0
    h_cols = set(range(n))
    # the 2 definitions should be useless, but avoid later errors in case of no
    # iterations is equal to 0
    tot_iter = 0
    tot_iter2 = 0
    print("-" * 20)
    print(f"tot_iter: expected [binom(n,r)] = {comb(n,r)}")
    if check_inner:
        tot_iter2 += 1
        print(f"tot_iter2: expected [binom(k,p)]= {comb(k,p)}")
    for tot_iter, isdstar_cols in enumerate(combinations(range(n), r)):
        isd_cols = sorted(tuple(h_cols - set(isdstar_cols)))
        h_rref = h.copy()
        syn_sig = syn.copy() if syn is not None else None
        # U is used just for double check
        if double_check:
            u = np.eye(r, dtype=np.ubyte)
        else:
            u = None
        rref(h_rref, isdstar_cols, syn_sig, u)
        h_right = h_rref[:, isdstar_cols]
        isiden = np.array_equal(h_right, iden)
        if isiden:
            tot_iden += 1
            if double_check:
                res = u @ h % 2
                try:
                    np.testing.assert_array_equal(res, h_rref)
                except:
                    print("***")
                    print(h)
                    print(isd_cols)
                    print(isdstar_cols)
                    print(u)
                    print(res)
        if not check_inner:
            continue
        # We proceed to extract independently from the identity matrix check,
        # simulating exactly the quantum circuit behaviour
        for tot_iter2, v_cols in enumerate(combinations(isd_cols, p)):
            sum_to_s = (h_rref[:, v_cols].sum(axis=1) + syn_sig) % 2
            sum_to_s_w = np.sum(sum_to_s)
            if sum_to_s_w == t - p:
                tot_correct_weight += 1
                if isiden:
                    tot_correct_weight_iden += 1
                # tot_correct_weight_cols.append((isd_cols, v_cols))

    tot_iter += 1
    print(f"tot_iter: real = {tot_iter}")
    if check_inner:
        tot_iter2 += 1
        print(f"tot_iter2: real = {tot_iter2}")

    print("-" * 20)
    print(f"# identity matrices")
    print(f"expected [.288 * r * r]: {.288*r*r}")
    print(f"real = {tot_iden}")

    # print("Some stats")
    print("-" * 20)
    print(f"% identities = tot_iden / tot_iter")
    print(f"expected: [prod_(i=1)(r)(1-1/2^i)] = .288")
    print(f"real: {tot_iden / tot_iter}")

    if check_inner:
        print("-" * 20)
        print(f"# Correct weights")
        print(f"expected ?= [binom(r,t-p)] {comb(r,t-p)}")
        print(
            f"real (independently of identity matrix) = {tot_correct_weight}")
        print(f"real (given matrix was identity) = {tot_correct_weight_iden}")
        print("-" * 20)
        print(f"% Correct weights")
        print(
            f"% total correct weights = tot_correct_weight / (tot_iter * tot_iter2): {tot_correct_weight / (tot_iter * tot_iter2)}"
        )
        print(
            f"% total correct weights identity = tot_correct_weight_iden / (tot_iter * tot_iter2): {tot_correct_weight_iden / (tot_iter * tot_iter2)}"
        )
    print("*" * 30)


def get_matrix(n, k, r, d, w, option: str):
    """Return matrix h (size r*k)"""
    if option == "random":
        return _gen_random_matrix_and_rank_check(r, k)
    elif option == "nonsys":
        return rcg.generate_parity_matrix_nonsystematic_for_hamming_from_r(5)
    elif option == "sys":
        return rch.get_isd_systematic_parameters(n, k, d, w)
    else:
        raise Exception("invalid choice")


def _gen_random_matrix_and_rank_check(r, k):
    rng = np.random.default_rng()
    # Discrete uniform distribution
    h = rng.integers(2, size=(r, k))
    rank = np.linalg.matrix_rank(h)
    while rank != r:
        h = rng.integers(2, size=(r, k))
        rank = np.linalg.matrix_rank(h)
    return h


def iden(h):
    """Only check the # of identities. In other words, from a given matrix h, we
compute all possible permutations of columns, apply RREF and check if the right
(or left, depending on the conventions) part is an identity matrix.
    """
    r, n = h.shape
    print(f"n {n} k {n-r} r {r} ")
    go(h, None, None, None, double_check=False, check_inner=False)


def iden_and_w(h, w, syndromes):
    """Check both # of identities and # of correct syndrome weights. Basically, we
    also check that, after RREF, the syndrome has the correct weight.

    """
    r, n = h.shape
    k = n - r
    # for p in reversed(range(w + 1)):
    # for p in range(1, 3):
    for p in range(w + 1):
        print(f"n {n} k {k} r {r}\nt {w} p {p}")
        go(h, w, p, syndromes[0], double_check=True, check_inner=True)


def main():
    #
    # r 4..6
    # h = rcg.generate_parity_matrix_nonsystematic_for_hamming_from_r(6)
    #
    # r, k
    # h = get_matrix(n=None, k=k, r=r, d=None, w=None, option="random")
    #
    # n, k, d, w = 7, 4, 3, 1
    n, k, d, w = 16, 11, 4, 1
    # n, k, d, w = 23, 12, 7, 3
    h, g, syndromes, errors, w, isHamming = get_matrix(n=n,
                                                       k=k,
                                                       d=d,
                                                       w=w,
                                                       r=None,
                                                       option="sys")
    # iden(h)
    iden_and_w(h, w, syndromes)


if __name__ == '__main__':
    main()
