import unittest
from library.matching_matrices.matching_matrix import matching_matrix


class TestMatchingMatrix(unittest.TestCase):

    def test_init_correct_defaults(self):

        # Arrange / Act
        m = matching_matrix(4)

        # Assert
        expected_rows = {0: {0: 1}, 1: {1: 1}, 2: {2: 1}, 3: {3: 1}}
        self.assertDictEqual(expected_rows, m.rows)

        expected_columns = {0: {0: 1}, 1: {1: 1}, 2: {2: 1}, 3: {3: 1}}
        self.assertDictEqual(expected_columns, m.columns)

        self.assertDictEqual({0: 1, 1: 1, 2: 1, 3: 1}, m.rtot)
        self.assertDictEqual({0: 1, 1: 1, 2: 1, 3: 1}, m.ctot)
        self.assertEqual(4, m.n)

        self.assertEqual(0, m.T)
        self.assertEqual(0, m.P)
        self.assertEqual(0, m.Q)

        self.assertDictEqual({}, m.update_A)
        self.assertDictEqual({}, m.update_B)

    def test_relabel_A_clusters_each_with_one_point(self):

        # set-up
        m = matching_matrix(10)

        i_1, i_2 = m.relabel_A(0, 1, 0)

        self.assertEqual(0, i_1)
        self.assertEqual(1, i_2)

        # there are 10 clusters 0-9 before the merge_cols
        # after the merge we create cluster 10 which should map to 1
        # in the row dictionary as we merge 0 into 1 arbitrarily taking the second
        # cluster to be the largest
        self.assertDictEqual({10: 1}, m.update_A)

    def test_relabel_A_clusters_first_with_2_second_with_1(self):

        # set-up
        m = matching_matrix(10)

        # merge 0 and 1 to create cluster 10 in step 0 and store it in column 1
        m.merge_rows(0, 1, 0)

        # merge 10 (stored in column 1) and 2 to create cluster 11 in step 1
        i_1, i_2 = m.relabel_A(10, 2, 1)

        self.assertEqual(2, i_1)  # cluster 2 is the smallest
        self.assertEqual(1, i_2)  # cluster 10 is the largest and is stored in column 1

        self.assertDictEqual({11: 1}, m.update_A)

    def test_relabel_A_clusters_first_with_1_second_with_2(self):

        # set-up
        m = matching_matrix(10)

        # merge 0 and 1 to create cluster 10 in step 0 and store it in column 1
        m.merge_rows(0, 1, 0)

        # merge 2 and 10 (stored in column 1) to create cluster 11 in step 1
        i_1, i_2 = m.relabel_A(10, 2, 1)

        self.assertEqual(2, i_1)  # cluster 2 is the smallest
        self.assertEqual(
            1, i_2
        )  # c cluster 10 is the largest and is stored in column 1

        self.assertDictEqual({11: 1}, m.update_A)

    def test_relabel_B_clusters_each_with_one_point(self):

        # set-up
        m = matching_matrix(10)

        j_1, j_2 = m.relabel_B(0, 1, 0)

        self.assertEqual(0, j_1)
        self.assertEqual(1, j_2)

        # there are 10 clusters 0-9 before the merge_cols
        # after the merge we create cluster 10 which should map to 1
        # in the row dictionary as we merge 0 into 1 arbitrarily taking the second
        # cluster to be the largest
        self.assertDictEqual({10: 1}, m.update_B)

    def test_relabel_B_clusters_first_with_2_second_with_1(self):

        # set-up
        m = matching_matrix(10)

        # merge 0 and 1 to create cluster 10 in step 0 and store it in column 1
        m.merge_columns(0, 1, 0)

        # merge 10 (stored in column 1) and 2 to create cluster 11 in step 1
        j_1, j_2 = m.relabel_B(10, 2, 1)

        self.assertEqual(2, j_1)  # cluster 2 is the smallest
        self.assertEqual(1, j_2)  # cluster 10 is the largest and is stored in column 1

        self.assertDictEqual({11: 1}, m.update_B)

    def test_relabel_B_clusters_first_with_1_second_with_2(self):

        # set-up
        m = matching_matrix(10)

        # merge 0 and 1 to create cluster 10 in step 0 and store it in column 1
        m.merge_columns(0, 1, 0)

        # merge 2 and 10 (stored in column 1) to create cluster 11 in step 1
        j_1, j_2 = m.relabel_B(10, 2, 1)

        self.assertEqual(2, j_1)  # cluster 2 is the smallest
        self.assertEqual(
            1, j_2
        )  # c cluster 10 is the largest and is stored in column 1

        self.assertDictEqual({11: 1}, m.update_B)

    def test_worked_example(self):

        # ================
        # -------A--------
        # ================
        #  i1, i2

        # [  3,  4 ]
        # [  1,  5 ]
        # [  0,  2 ]
        # [  6,  7 ]

        # ================
        # -------B--------
        # ================
        #  j1, j2, j_size

        # [  3,  4,  2]
        # [  0,  2,  2]
        # [  1,  5,  3]
        # [  6,  7,  5]

        # Arrange
        m = matching_matrix(5)

        """ Merge 0 in A
        0, 1, 2, 3, 4
    0 [ 1, 0, 0, 0, 0] 1
    1 [ 0, 1, 0, 0, 0] 1
    2 [ 0, 0, 1, 0, 0] 1
    5 [ 0, 0, 0, 1, 1] 2
      1  1  1  1  1

    """
        m.merge_rows(i_1=3, i_2=4, k=0)

        self.assertEqual(0, m.T)
        self.assertEqual(1, m.P)
        self.assertEqual(0, m.Q)

        """ Merge 0 in B
      0, 1, 2, 5
    0 [ 1, 0, 0, 0 ] 1
    1 [ 0, 1, 0, 0 ] 1
    2 [ 0, 0, 1, 0 ] 1
    5 [ 0, 0, 0, 2 ] 2
      1, 1, 1, 2
    """

        m.merge_columns(j_1=3, j_2=4, k=0)

        self.assertEqual(1, m.T)
        self.assertEqual(1, m.P)
        self.assertEqual(1, m.Q)

        """ Merge 1 in A
        0, 1, 2, 5
    0 [ 1, 0, 0, 0 ] 1
    2 [ 0, 0, 1, 0 ] 1
    6 [ 0, 1, 0, 2 ] 3
        1, 1, 1, 2
    """
        m.merge_rows(i_1=1, i_2=5, k=1)

        self.assertEqual(1, m.T)
        self.assertEqual(3, m.P)
        self.assertEqual(1, m.Q)

        """ Merge 1 in B
        6  1  5
    0 [ 1, 0, 0 ] 1
    2 [ 1, 0, 0 ] 1
    6 [ 0, 1, 2 ] 3
        2, 1, 2
    """
        m.merge_columns(j_1=0, j_2=2, k=1)

        self.assertEqual(1, m.T)
        self.assertEqual(3, m.P)
        self.assertEqual(2, m.Q)

        """ Merge 2 in A
        6  1  5
    7 [ 2, 0, 0 ] 2
    6 [ 0, 1, 2 ] 3
        2, 1, 2
    """
        m.merge_rows(i_1=0, i_2=2, k=2)

        self.assertEqual(2, m.T)
        self.assertEqual(4, m.P)
        self.assertEqual(2, m.Q)

        """ Merge 2 in B
        6  7
    7 [ 2, 0 ] 2
    6 [ 0, 3 ] 3
        2, 3
    """
        m.merge_columns(j_1=1, j_2=5, k=2)

        self.assertEqual(4, m.T)
        self.assertEqual(4, m.P)
        self.assertEqual(4, m.Q)

        """ Merge 3 in A
        6  7
    8 [ 2, 3 ] 5
        2, 3
    """

        m.merge_rows(i_1=6, i_2=7, k=3)

        self.assertEqual(4, m.T)
        self.assertEqual(10, m.P)
        self.assertEqual(4, m.Q)

        """ Merge 3 in B
        8
    8 [ 5 ] 5
        5
    """
        m.merge_columns(j_1=6, j_2=7, k=3)

        self.assertEqual(10, m.T)
        self.assertEqual(10, m.P)
        self.assertEqual(10, m.Q)


if __name__ == "__main__":
    unittest.main()
