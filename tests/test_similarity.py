import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from library.similarity import similarity_metrics
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score


class TestSimilarityMetrics(unittest.TestCase):

    def setUp(self):

        self.large_A = np.array(
            [
                [1.0, 6.0, 0.60, 2.0],
                [8.0, 9.0, 0.85, 2.0],
                [4.0, 5.0, 1.50, 2.0],
                [10.0, 12.0, 1.56, 4.0],
                [7.0, 13.0, 2.42, 5.0],
                [3.0, 11.0, 2.60, 3.0],
                [14.0, 15.0, 2.63, 8.0],
                [2.0, 16.0, 4.14, 9.0],
                [0.0, 17.0, 5.14, 10.0],
            ]
        )

        self.large_B = np.array(
            [
                [1.0, 6.0, 0.60, 2.0],
                [8.0, 9.0, 0.85, 2.0],
                [4.0, 5.0, 1.50, 2.0],
                [10.0, 12.0, 2.21, 4.0],
                [3.0, 11.0, 3.00, 3.0],
                [7.0, 13.0, 3.07, 5.0],
                [2.0, 15.0, 4.73, 6.0],
                [0.0, 14.0, 4.76, 4.0],
                [16.0, 17.0, 7.68, 10.0],
            ]
        )

        # after the first merge the flat clusterings should be identical
        # after the second merge we should have the following matrix
        # [[2, 1],
        #  [1, 0]]
        # T = 1, P = 3, Q = 3, N = 6

        self.small_A = np.array(
            [[0.0, 2.0, 0.11, 2.0], [1.0, 4.0, 0.23, 3.0], [3.0, 5.0, 0.24, 4.0]]
        )

        self.small_B = np.array(
            [[0.0, 2.0, 0.11, 2.0], [3.0, 4.0, 0.25, 3.0], [1.0, 5.0, 0.27, 4.0]]
        )

    def test_compare_example_with_sklearn(self):

        # Act
        metrics = similarity_metrics(self.large_A, self.large_B)

        ar_similarity = metrics.adjusted_rand()
        fm_similarity = metrics.fowlkes_mallows()

        ar_sklearn = []
        fm_sklearn = []

        for i in range(9, 1, -1):

            fcluster_a = fcluster(self.large_A, i, "maxclust")
            fcluster_b = fcluster(self.large_B, i, "maxclust")

            ar = adjusted_rand_score(fcluster_a, fcluster_b)
            fm = fowlkes_mallows_score(fcluster_a, fcluster_b)

            ar_sklearn.append(ar)
            fm_sklearn.append(fm)

        # Assert
        assert_almost_equal(ar_similarity, ar_sklearn)
        assert_almost_equal(fm_similarity, fm_sklearn)

    def test_rand(self):

        # Act
        metrics = similarity_metrics(self.small_A, self.small_B)

        # Assert

        # identical should be 1
        self.assertEqual(1, metrics.rand()[0])  #

        # If T = 1, P = 3, Q = 3, N = 6 then
        # R =  (6 - 3 - 3 + 2 * 1)/6 = 1/3
        self.assertEqual(1 / 3, metrics.rand()[1])

    def test_fowlkes_mallows(self):

        # Act
        metrics = similarity_metrics(self.small_A, self.small_B)

        # Assert

        # identical should be 1
        self.assertEqual(1, metrics.fowlkes_mallows()[0])

        # If T = 1, P = 3, Q = 3, N = 6 then
        # FM = 1 / np.sqrt(3*3) = 1/3
        self.assertEqual(1 / 3, metrics.fowlkes_mallows()[1])

    def test_adjusted_rand(self):

        # Act
        metrics = similarity_metrics(self.small_A, self.small_B)

        # Assert

        # identical should be 1
        self.assertEqual(1, metrics.adjusted_rand()[0])

        # If T = 1, P = 3, Q = 3, N = 6 then
        # AR = 2 * (6 * 1 - 3 * 3) / (6 * (3 + 3) - 2 * 3 * 3) = -6/18 = -1/3
        self.assertEqual(-1 / 3, metrics.adjusted_rand()[1])

    def test_similarity_single_indices(self):

        metrics = similarity_metrics(self.large_A, self.large_B)

        assert_equal(metrics.adjusted_rand(), metrics.get_index("ar")["ar"])
        assert_equal(
            metrics.adjusted_rand(), metrics.get_index("adjustedrand")["adjustedrand"]
        )
        assert_equal(
            metrics.adjusted_rand(), metrics.get_index("adjusted_rand")["adjusted_rand"]
        )

        assert_equal(metrics.fowlkes_mallows(), metrics.get_index("b")["b"])
        assert_equal(metrics.fowlkes_mallows(), metrics.get_index("fm")["fm"])
        assert_equal(
            metrics.fowlkes_mallows(),
            metrics.get_index("fowlkesmallows")["fowlkesmallows"],
        )
        assert_equal(
            metrics.fowlkes_mallows(),
            metrics.get_index("fowlkes_mallows")["fowlkes_mallows"],
        )

        assert_equal(metrics.rand(), metrics.get_index("r")["r"])
        assert_equal(metrics.rand(), metrics.get_index("rand")["rand"])

    def test_similarity_multiple_indices(self):

        metrics = similarity_metrics(self.large_A, self.large_B)

        output = metrics.get_index(["ar", "fm", "r"])

        assert_equal(metrics.adjusted_rand(), output["ar"])
        assert_equal(metrics.fowlkes_mallows(), output["fm"])
        assert_equal(metrics.rand(), output["r"])


if __name__ == "__main__":
    unittest.main()
