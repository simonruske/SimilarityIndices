from collections import Counter


class matching_matrix_pairs:
    """
    Class used to store a matching matrix. The matching matrix is used
    to store information about two clusterings of a set of :math:n
    objects. We call the first clustering :math:`A` and the second :
    math:`B`. Each row represents a cluster :math:`i` from the
    clustering :math:`A`. Each column represents a cluster :math:`j`
    from the clustering :math:`B`. Each element of the matching matrix
    will then be the number of objects in the cluster :math:`i` that are
    also in the cluster :math:`j`.

    This version of the matching matrix is designed to be used to
    compare a single clustering to a hierarchy. Hence will take input
    of a particular clustering as well as n the number of objects in
    the clustering.

    Parameters
    ----------
    n : integer
        The number of objects in the clustering

    sol : ndarray
        The clustering to be compared to the hierarchy.

    """

    def __init__(self, sol, n, *args, **kwargs):

        self.rows = {i: {j: 1} for i, j in enumerate(sol)}
        self.rtot = {i: 1 for i, j in enumerate(sol)}
        self.n = n
        c = Counter(sol)
        self.T = 0
        self.P = 0
        self.Q = (sum([i**2 for i in c.values()]) - n) // 2
        self.update_A = {}

    def merge_rows(self, i_1, i_2, k):

        if i_1 >= self.n:
            i_1 = self.update_A.pop(i_1)

        if i_2 >= self.n:
            i_2 = self.update_A.pop(i_2)

        if self.rtot[i_1] >= self.rtot[i_2] and i_1 >= self.n and i_2 >= self.n:
            i_1, i_2 = i_2, i_1

        self.update_A[k + self.n] = i_2

        rtot1, rtot2 = self.rtot.pop(i_1), self.rtot[i_2]
        self.rtot[i_2] = rsum = rtot1 + rtot2
        self.P += sum([rsum**2, -(rtot1**2), -(rtot2**2)]) // 2

        r1, r2, st = self.rows.pop(i_1), self.rows[i_2], 0

        for elem in r1:

            if elem not in r2:
                r2[elem] = r1[elem]

            else:
                value_1 = r1[elem]
                value_2 = r2[elem]
                value_new = value_1 + value_2
                r2[elem] = value_new
                st += sum((value_new**2, -(value_1**2), -(value_2**2)))

        self.T += st // 2
        self.rows[i_2] = r2
