def rref(mat, idx_cols: tuple, syn=None, u=None):
    """Compute the RREF of the matrix mat (producing mat_rref), trying to bring the
    rightmost square matrix to identity. Rightmost square matrix columns are
    selected through idx_cols.

    If syn is not None, it also applies the same sequence of transformation of
    RREF to it.

    If u is not None, it also applies the same sequence of transformation to
    it. To be meaningful, u should be an identity matrix. In this way, at the
    end, it'll store all the sequence of operations applied to mat. In this
    case, we have that u @ mat = mat_rref

    """

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
            if u is not None:
                _sum_rows(u[row1, :], u[row2, :], fake_swaps[-1])
            if syn is not None:
                syn[row1] = _mcnot_as_logic(syn[row1], syn[row2],
                                            fake_swaps[-1])

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
            if u is not None:
                _sum_rows(u[row2, :], u[row1, :], col_adds[-1])
            if syn is not None:
                syn[row2] = _mcnot_as_logic(syn[row2], syn[row1], col_adds[-1])


def _sum_rows(dst_row, src_row, ctrl):
    for i, (dst_cell, src_cell) in enumerate(zip(dst_row, src_row)):
        dst_row[i] = _mcnot_as_logic(dst_cell, src_cell, ctrl)


def _mcnot_as_logic(dst, *ctrls):
    # reduce doesn't short-circuit
    # return dst ^ functools.reduce(operator.and_, ctrls)
    return dst ^ all(ctrls)
