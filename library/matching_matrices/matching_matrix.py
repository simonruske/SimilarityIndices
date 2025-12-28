class matching_matrix:
    """
    Class used to store a matching matrix. The matching matrix is used
    to store information about two clusterings of a set of :math:n
    objects. We call the first clustering :math:`A` and the second :
    math:`B`. Each row represents a cluster :math:`i` from the
    clustering :math:`A`. Each column represents a cluster :math:`j`
    from the clustering :math:`B`. Each element of the matching matrix
    will then be the number of objects in the cluster :math:`i` that are
    also in the cluster :math:`j`.

    Initially for both clusterings we let each object be in a cluster on
    it's own so the matching matrix is an identity matrix of size n. The
    merge function is then used to successively merge two clusters from
    :math:`A` and :math:`B` until there is only one cluster left in each
    and the matching matrix is :math:`[n]`

    Parameters
    ----------
    n : integer
        An integer for the size of the initial matching matrix.

    """

    def __init__(self, n):

        # The matrix itself
        self.rows = {x: {x: 1} for x in range(n)}
        self.columns = {x: {x: 1} for x in range(n)}

        # Row and column totals
        self.rtot = {x: 1 for x in range(n)}
        self.ctot = {x: 1 for x in range(n)}
        self.n = n

        # TPQ
        self.T = 0
        self.P = 0
        self.Q = 0

        # Dictionaries used for the relabelling procedure
        self.update_A = {}
        self.update_B = {}

    def relabel_A(self, i_1, i_2, k):
        """

        Relabelling procedure for A. An entry is added to update_A for each
        newly formed cluster mapping the index `n+k` to the index of the largest of the
        two clusters that merged in order to create the newly formed cluster. This relabeling
        is to decrese the number of insertions into a dictionary when cluster merge and consequently
        decreases execution time. If both clusters contain only one point the original index is returned.

        Parameters
        ----------
        i_1 : int
             The label of the first cluster to be merged in
             hierarhical clustering A

        i_2 : int
            The label of the second cluster to be merged in
            hierarchical clustering A

        k: int
            The number of merges that have already taken place
            in the hierarchical clustering A.

        Returns
        -------
        i_1 : int
            Index in the row dictionary of the cluster with the least number of points.

        i_2 : int
            Index in the row dictionary of the cluster with the largest number of points.

        """

        if i_1 >= self.n:
            i_1 = self.update_A.pop(i_1)

        if i_2 >= self.n:
            i_2 = self.update_A.pop(i_2)

        if self.rtot[i_1] > self.rtot[i_2]:
            i_1, i_2 = i_2, i_1

        self.update_A[k + self.n] = i_2

        return i_1, i_2

    def relabel_B(self, j_1, j_2, k):
        """

        Relabelling procedure for B. An entry is added to update_B for each
        newly formed cluster mapping the index `n+k` to the index of the largest of the
        two clusters that merged in order to create the newly formed cluster. This relabeling
        is to decrese the number of insertions into a dictionary when cluster merge and consequently
        decreases execution time. If both clusters contain only one point the original index is returned.

        Parameters
        ----------

        j_1 : int
             The label of the first cluster to be merged in
             hierarhical clustering B

        j_2 : int
            The label of the second cluster to be merged in
            hierarchical clustering B

        k: int
            The number of merges that have already taken place
            in the hierarchical clustering B.

        Returns
        -------
        i_1 : int
            Index in the row dictionary of the cluster with the least number of points.

        i_2 : int
            Index in the row dictionary of the cluster with the largest number of points.

        """

        if j_1 >= self.n:
            j_1 = self.update_B.pop(j_1)

        if j_2 >= self.n:
            j_2 = self.update_B.pop(j_2)

        if self.ctot[j_1] > self.ctot[j_2]:
            j_1, j_2 = j_2, j_1

        self.update_B[k + self.n] = j_2

        return j_1, j_2

    def update_row_totals_and_P(self, i_1, i_2):
        """

        Parameters
        ----------

        i_1 : int
             The label of the first cluster to be merged in
             hierarhical clustering A

        i_2 : int
            The label of the second cluster to be merged in
            hierarchical clustering A
        """

        rtot1, rtot2 = self.rtot.pop(i_1), self.rtot[i_2]
        self.rtot[i_2] = rtot1 + rtot2
        self.P += rtot1 * rtot2

    def update_column_totals_and_Q(self, j_1, j_2):
        """

        Parameters
        ----------

        j_1 : int
             The label of the first cluster to be merged in
             hierarhical clustering B

        j_2 : int
            The label of the second cluster to be merged in
            hierarchical clustering B

        """

        ctot1, ctot2 = self.ctot.pop(j_1), self.ctot[j_2]
        self.ctot[j_2] = ctot1 + ctot2
        self.Q += ctot1 * ctot2

    def update_row_dictionary_and_T(self, i_1, i_2):
        """
        Procedure used to merge the clusters in A, the variable 'st'
        is used to store the increase in T. The column version of
        the matrix is updated at the same time. We add the smallest
        row r1 to the largest row r2.

        For every element in the smallest row, if it's not in the other row then
        add it, if it is then add it on and update the value of T. In
        either case update the column dictionary with the change.

        Parameters
        ----------

        i_1 : int
            The label of the first cluster to be merged in
            hierarhical clustering A

        i_2 : int
            The label of the second cluster to be merged in
            hierarchical clustering A

        """

        r1, r2 = self.rows.pop(i_1), self.rows[i_2]

        for elem in r1:

            if elem in r2:
                value_1 = self.columns[elem].pop(i_1)
                value_2 = self.columns[elem][i_2]
                value_new = value_1 + value_2
                r2[elem] = self.columns[elem][i_2] = value_new
                self.T += value_1 * value_2

            else:
                r2[elem] = self.columns[elem][i_2] = self.columns[elem].pop(i_1)

    def update_column_dictionary_and_T(self, j_1, j_2):
        """
        Procedure used to merge the clusters in A, the variable 'st'
        is used to store the increase in T. The column version of
        the matrix is updated at the same time. We add the smallest
        row r1 to the largest row r2.

        For every element in the smallest row, if it's not in the other row then
        add it, if it is then add it on and update the value of T. In
        either case update the column dictionary with the change.

        Parameters
        ----------

        j_1 : int
             The label of the first cluster to be merged in
             hierarhical clustering B

        j_2 : int
            The label of the second cluster to be merged in
            hierarchical clustering B

        """

        c1, c2 = self.columns.pop(j_1), self.columns[j_2]

        for elem in c1:

            if elem in c2:
                value_1 = self.rows[elem].pop(j_1)
                value_2 = self.rows[elem][j_2]
                value_new = value_1 + value_2
                c2[elem] = self.rows[elem][j_2] = value_new
                self.T += value_1 * value_2

            else:
                c2[elem] = self.rows[elem][j_2] = self.rows[elem].pop(j_1)

    def merge_rows(self, i_1, i_2, k):
        """

        Parameters
        ----------

        i_1 : int
             The label of the first cluster to be merged in
             hierarhical clustering A

        i_2 : int
            The label of the second cluster to be merged in
            hierarchical clustering A

        k: int
            The number of merges that have already taken place
            in the hierarchical clustering A.

        """

        i_1, i_2 = self.relabel_A(i_1, i_2, k)
        self.update_row_totals_and_P(i_1, i_2)
        self.update_row_dictionary_and_T(i_1, i_2)

    def merge_columns(self, j_1, j_2, k):
        """

        Parameters
        ----------

        j_1 : int
             The label of the first cluster to be merged in
             hierarhical clustering B

        j_2 : int
            The label of the second cluster to be merged in
            hierarchical clustering B

        k: int
            The number of merges that have already taken place
            in the hierarchical clustering B.

        """

        j_1, j_2 = self.relabel_B(j_1, j_2, k)
        self.update_column_totals_and_Q(j_1, j_2)
        self.update_column_dictionary_and_T(j_1, j_2)

    def merge(self, i_1, i_2, j_1, j_2, k):
        """
        Parameters
        ----------

        i_1, i_2 : int
            The labels of the two clusters to be merged in the
            clustering A.

        j_1, j_2 : integers
            The labels of the two clusters to be merged in the
            clustering B.

        k : integer
            Number of merges that have taken place before this merge

        """

        self.merge_rows(i_1, i_2, k)
        self.merge_columns(j_1, j_2, k)
        return (self.T, self.P, self.Q)
