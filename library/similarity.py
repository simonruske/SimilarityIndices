import numpy as np

from library.matching_matrices.matching_matrix import matching_matrix
from scipy.cluster.hierarchy import is_valid_linkage


class similarity_metrics:
    """
    Class used to get metrics to compare two hierarchical clusterings

    Parameters
    ----------
    A : ndarray
        A :math:`(n-1)` by 4 matrix encoding the linkage
        (hierarchical clustering).  See ``linkage`` documentation
        for more information on its form.
    B : A second :math:`(n-1)` by 4 matrix encoding the linkage
      (hierarchical clustering).

    """

    def __init__(self, A, B):

        self.TPQ_linkages(A, B)

    def TPQ_linkages(self, A, B):
        """
        Calculates statistics on two hierarchical clusterings of the same set of objects
        which can be used to calculate indices to compare two hierarchical clusterings

        Parameters
        ----------
        A : ndarray
            A :math:`(n-1)` by 4 matrix encoding the linkage
            (hierarchical clustering).  See ``linkage`` documentation
            for more information on its form.
        B : A second :math:`(n-1)` by 4 matrix encoding the linkage
            (hierarchical clustering).

        Returns
        -------
        T : ndarray
            A vector of size :math:'n-1' where the ``k``'th element
            contains the number of pairs of objects placed into the
            same cluster for both hierarchical clusterings after the
            clusters in the ``k``'th row have been merged.
        P : ndarray
            A vector of size :math:'n-1' where the ``k``'th element
            contains the number of pairs of objects placed into the
            same cluster after the clusters in the ``k``th row
            for the hierarchical clustering A only.
        Q : ndarray
            A vector of size :math:'n-1' where the ``k``'th element
            contains the number of pairs of objects placed into the
            same cluster after the clusters in the ``k``th row
            for the hierarchical clustering B only.
        """

        # Convert to array if not already.
        A = np.array(A, "double")
        B = np.array(B, "double")

        # Performs checks
        is_valid_linkage(A, throw=True)
        is_valid_linkage(B, throw=True)

        n = len(A) + 1
        n2 = len(B) + 1

        if n != n2:
            raise ValueError("The hierarchical clusterings must be of the same size")

        self.T = np.zeros(n - 2)
        self.P = np.zeros(n - 2)
        self.Q = np.zeros(n - 2)
        self.n = n

        # Creates a new matching matrix (identity of size n)
        m = matching_matrix(n)

        # Merges the required clusters as specified by the input files
        for k, (rows_A, rows_B) in enumerate(zip(A, B)):
            if k != n - 2:
                self.T[k], self.P[k], self.Q[k] = m.merge(
                    rows_A[0], rows_A[1], rows_B[0], rows_B[1], k
                )

    def get_index(self, index):
        """
        Calculates an index to determine how similar the two
        hierarchical clusterings A and B are.

        The following indices are available.

          * index='R, Rand
            .. math::
               R = \\frac{n(n-1)-2(2 T_k - P_k - Q_k)}{n(n-1)}

          * index='AR', Adjusted Rand
            .. math::
               AR = \\frac{2T_k-\\frac{P_k Q_k}{n(n-1)}}{P_k+Q_k - \\frac{P_k+Q_k}{n(n-1)}}

          * index='B', Fowlkes and Mallows index
            .. math::
               B = \\frac{T_k}{\\sqrt{P_k Q_k}}


        With the exception of the Adjusted Rand each of the indices are defined on
        the interval :math:`[0,1]` where values close to 1 indicate strong
        similarity and values close to 0 indicate lack of similairity. The same is
        true for the Adjusted Rand index except it is possible that the index can
        take on values in the interval :math:`[-1,0)` when the value of the Rand
        index is less than it's expected value.

        Parameters
        ----------

        index : string or list
            Either 'AR', 'R', 'B', indicating which index you would
            like to use. In the case of list a list of the above

        Returns
        -------

        result : ndarray or float

            With the exception of the S-index the result with be
            a vector of size :math:'n-1' where the :math:``k``'th element
            contains an index indicating how similar the clusterings A
            and B are after the :math: . For the S-index the result will
            be a single index.

        """

        if isinstance(index, str):
            indices = [index]

        elif isinstance(index, list):
            indices = index

        else:
            raise ValueError("Index must either be a string or a list of indices")

        output = {}

        for index in indices:

            index = index.lower()

            if index in ["r", "rand"]:
                output[index] = self.rand()

            elif index in ["ar", "adjustedrand", "adjusted_rand"]:
                output[index] = self.adjusted_rand()

            elif index in ["b", "fm", "fowlkesmallows", "fowlkes_mallows"]:
                output[index] = self.fowlkes_mallows()

        return output

    def rand(self):
        """
        Calculates the rand score

        math::
          R_k = \\frac{n(n-1)-2(2 T_k - P_k - Q_k)}{n(n-1)}
        """

        N = self.n * (self.n - 1) // 2
        return (N - self.P - self.Q + 2 * self.T) / N

    def adjusted_rand(self):
        """
        Calculates the adjusted rand score

        math::
          AR = \\frac{2T_k-\\frac{P_k Q_k}{n(n-1)}}{P_k+Q_k - \\frac{P_k+Q_k}{n(n-1)}}
        """

        N = self.n * (self.n - 1) // 2
        return (
            2
            * (N * self.T - self.P * self.Q)
            / (N * (self.P + self.Q) - 2 * self.P * self.Q)
        )

    def fowlkes_mallows(self):
        """
        Calculates the Fowlkes and Mallows index
        math::
          B = \\frac{T_k}{\\sqrt{P_k Q_k}}
        """

        return self.T / np.sqrt(self.P * self.Q)
