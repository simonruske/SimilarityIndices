from .matching_matrices import matching_matrix_proportion, matching_matrix_pairs
from collections import Counter
import numpy as np


def proportion(A, B):

    n = len(A) + 1
    prop = np.zeros(n + 1)
    m = matching_matrix_proportion(B, n)

    for k, rows_A in enumerate(A):
        m.merge_rows(rows_A[0], rows_A[1], rows_A[3], k)
        maximums = m.maximums
        where = m.where
        total = 0
        for index in Counter(where).keys():
            total += max(maximums[where == index])
            prop[len(m.rows)] = float(total) / m.n

    return prop


def TPQ_known(A, B, num=None):

    n = len(A) + 1
    T = np.zeros(n - 1)
    P = np.zeros(n - 1)
    Q = np.zeros(n - 1)

    # Creates a new matching matrix (identity of size n)
    m = matching_matrix_pairs(B, n)

    # Merges the required clusters as specified by the input files
    for k, rows_A in enumerate(A):
        m.merge_rows(rows_A[0], rows_A[1], rows_A[3], k)
        T[k], P[k], Q[k] = m.T, m.P, m.Q

    return T[:-1], P[:-1], Q[:-1]
