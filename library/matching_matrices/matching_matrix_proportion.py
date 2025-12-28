from collections import Counter
import numpy as np


class matching_matrix_proportion:

    def __init__(self, sol, n, *args, **kwargs):

        self.rows = {i: {int(j): 1} for i, j in enumerate(sol)}
        self.n = n
        self.k = len(Counter(sol))
        self.update_A = {}
        self.maximums = np.zeros(self.k)
        self.where = np.zeros(self.k)

    def merge_rows(self, i_1, i_2, k):

        if i_1 >= self.n:
            i_1 = self.update_A.pop(i_1)

        if i_2 >= self.n:
            i_2 = self.update_A.pop(i_2)

        if (
            len(self.rows[i_1]) >= len(self.rows[i_2])
            and i_1 >= self.n
            and i_2 >= self.n
        ):
            i_1, i_2 = i_2, i_1

        self.update_A[k + self.n] = i_2

        r1, r2 = self.rows.pop(i_1), self.rows[i_2]

        # If the maximum is moving then update it's location
        if i_1 in self.where:
            self.where[np.where(self.where == i_1)] = i_2

        for elem in r1:

            if elem not in r2:
                r2[elem] = r1[elem]

            else:
                value_1 = r1[elem]
                value_2 = r2[elem]
                value_new = value_1 + value_2
                r2[elem] = value_new
                if value_new > self.maximums[elem]:
                    self.maximums[elem] = value_new
                    self.where[elem] = i_2

        self.rows[i_2] = r2

    def to_dense(self):

        m = len(self.rows)
        n = self.k
        M = np.zeros((m, n))
        for i, row in enumerate(self.rows):
            for j, col in self.rows[row].items():
                M[i, j] = self.rows[row][j]
        return M
