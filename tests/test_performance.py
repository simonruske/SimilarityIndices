import unittest
import numpy as np
from fastcluster import linkage
from time import perf_counter
from library.similarity import similarity_metrics
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score


class Performance(unittest.TestCase):

    def test_adjusted_rand_performance(self):

        # Arrange
        n = 100
        np.random.seed(seed=8455624)
        x = np.random.normal(n, 2, (n, 2))
        A = linkage(x, "centroid")
        B = linkage(x, "ward")

        # Act

        similarity_times = []
        sklearn_times = []
        fcluster_times = []

        for _ in range(100):

            start = perf_counter()

            metrics = similarity_metrics(A, B)
            ar_similarity = metrics.adjusted_rand()

            end = perf_counter()

            similarity_times.append(end - start)

            ar_sklearn = []

            sklearn_time = 0
            fcluster_time = 0

            actual_excluded_results = 0
            actual_included_results = 0
            for i in range(n - 1, 1, -1):

                start = perf_counter()

                fcluster_a = fcluster(A, i, "maxclust")
                fcluster_b = fcluster(B, i, "maxclust")

                end = perf_counter()

                fcluster_time += end - start

                start = perf_counter()

                ar = adjusted_rand_score(fcluster_a, fcluster_b)

                end = perf_counter()

                sklearn_time += end - start

                # fcluster takes maxclust rather than an exact number of clusters
                # most of the time it will create exactly maxclust, but for the occasions
                # that it doesn't the results are are not comparable so ignore them
                if (len(np.unique(fcluster_a)) != i) or (
                    len(np.unique(fcluster_b)) != i
                ):
                    actual_excluded_results += 1
                    ar_sklearn.append(ar_similarity[len(ar_sklearn)])

                else:
                    actual_included_results += 1
                    ar_sklearn.append(ar)

            sklearn_times.append(sklearn_time)
            fcluster_times.append(fcluster_time)

            ar_sklearn = np.array(ar_sklearn)

            # Assert
            self.assertEqual(len(ar_sklearn), len(ar_similarity))
            np.testing.assert_almost_equal(ar_similarity, ar_sklearn)

            # The number of excluded results should remain constant
            # barring changes to fcluster implementation.
            expected_number_of_excluded_results = 3
            self.assertEqual(
                expected_number_of_excluded_results, actual_excluded_results
            )
            self.assertEqual(
                n - 2 - expected_number_of_excluded_results, actual_included_results
            )

        # Write results to a temporary file
        average_similarity_time = np.average(similarity_times)
        average_sklearn_time = np.average(sklearn_times)

        with open(
            "performance_results.txt",
            mode="w",
        ) as temp_file:
            temp_file.write(f"Similarity average time: {average_similarity_time}\n")
            temp_file.write(f"Sklearn average time: {average_sklearn_time}\n")
            temp_file.write(f"FCluster average time: {np.average(fcluster_times)}\n")

        assert (
            average_similarity_time * 10 < average_sklearn_time
        )  # The performance should be at least 10x better


if __name__ == "__main__":
    unittest.main()
