from itertools import combinations

import numpy as np
from isdclassic.utils import rectangular_codes_hardcoded as rch

# from copy import deepcopy
# import functools
# import operator


def go(h, t, p, syn, double_check=False):
    assert t - p >= 0
    r, n = h.shape
    k = n - r
    iden = np.eye(r)
    tot_iden = 0
    tot_correct_weight = 0
    tot_correct_weight_iden = 0
    h_cols = set(range(n))
    # tot_correct_weight_cols = []
    for tot_iter, isdstar_cols in enumerate(combinations(range(n), r)):
        isd_cols = sorted(tuple(h_cols - set(isdstar_cols)))
        h_rref = h.copy()
        syn_sig = syn.copy()
        # U was used just for double check
        # u = np.eye(r, dtype=np.ubyte)
        # _rref(h_rref, syn_sig, u, isdstar_cols)
        _rref(h_rref, syn_sig, isdstar_cols)
        isiden = False
        if np.array_equal(h_rref[:, isdstar_cols], iden):
            tot_iden += 1
            isiden = True
            # if double_check:
            #     try:
            #         np.testing.assert_array_equal(u @ h % 2, h_rref)
            #     except:
            #         print("***")
            #         print(h)
            #         print(isd_cols)
            #         print(isdstar_cols)
            #         print(u)
            #         bla = u @ h % 2
            #         print(bla)
            #         input("a")

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
    tot_iter2 += 1
    print(f"tot_iter = binom(n,r) {tot_iter}")
    print(f"tot_iter2 = binom(k,p) {tot_iter2}")
    print(f"identity matrices {tot_iden}")
    print(
        f"correct weights (independently of identity matrix) {tot_correct_weight}"
    )
    print(
        f"correct weights (given matrix was identity) {tot_correct_weight_iden}"
    )
    print("Some stats")
    print(f"% total identities = tot_iden / tot_iter: {tot_iden / tot_iter}")
    print(
        f"% total correct weights = tot_correct_weight / (tot_iter * tot_iter2): {tot_correct_weight / (tot_iter * tot_iter2)}"
    )
    print(
        f"% total correct weights identity = tot_correct_weight_iden / (tot_iter * tot_iter2): {tot_correct_weight_iden / (tot_iter * tot_iter2)}"
    )
    print("*" * 30)


def _rref(mat, syn, idx_cols: tuple):
    # def _rref(mat, syn, u, idx_cols: tuple):
    fake_swaps = []
    col_adds = []
    # square matrix
    steps = 0
    for row1, col1 in enumerate(idx_cols):
        # Fake-swap the row if the pivot is 0
        for row2_add, col2 in enumerate(idx_cols[row1 + 1:]):
            row2 = row1 + 1 + row2_add
            steps += 1
            if mat[row1][col1] == 0:
                fake_swaps.append(1)
            else:  # == 1
                fake_swaps.append(0)
            _sum_rows(mat[row1, :], mat[row2, :], fake_swaps[-1])
            # _sum_rows(u[row1, :], u[row2, :], fake_swaps[-1])
            syn[row1] = _mcnot_as_logic(syn[row1], syn[row2], fake_swaps[-1])

        # Transform each row2 element below row1 into 0
        for row2, col2 in enumerate(idx_cols):
            if row2 == row1:
                continue
            steps += 1
            if mat[row2, col1] == 1:
                col_adds.append(1)
            else:
                col_adds.append(0)
            _sum_rows(mat[row2, :], mat[row1, :], col_adds[-1])
            # _sum_rows(u[row2, :], u[row1, :], col_adds[-1])
            syn[row2] = _mcnot_as_logic(syn[row2], syn[row1], col_adds[-1])
    # return u


def _sum_rows(dst_row, src_row, ctrl):
    for i, (dst_cell, src_cell) in enumerate(zip(dst_row, src_row)):
        dst_row[i] = _mcnot_as_logic(dst_cell, src_cell, ctrl)


def _mcnot_as_logic(dst, *ctrls):
    # reduce doesn't short-circuit
    # return dst ^ functools.reduce(operator.and_, ctrls)
    return dst ^ all(ctrls)


def _get_error(isd_cols, v_cols, sum_to_s):
    """ Used only in old double check"""
    k = len(isd_cols)
    e_hat = np.concatenate((np.zeros(k), sum_to_s))
    v_cols_idxs = [isd_cols.index(v_col) for v_col in v_cols]
    for j in v_cols_idxs:
        e_hat[j] = 1
    return e_hat


def main():
    n, k, d, w = 23, 12, 7, 3
    # n, k, d, w = 16, 11, 4, 1
    # n, k, d, w = 7, 4, 3, 1
    h, g, syndromes, errors, w, isHamming = rch.get_isd_systematic_parameters(
        n, k, d, w)
    # for p in range(1, 3):
    for p in range(w + 1):
        # for p in reversed(range(w + 1)):
        print(f"n {n} k {k} t {w} p {p}")
        go(h, w, p, syndromes[0], double_check=False)


if __name__ == '__main__':
    main()
