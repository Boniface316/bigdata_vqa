from typing import Callable, List, Tuple, Union

from ._coreset import Coreset

import numpy as np


class BFL2(Coreset):
    def __init__(
        self,
        raw_data: np.ndarray,
        number_of_sampling_for_centroids: int,
        coreset_size: int,
        number_of_coresets_to_evaluate: int,
        sampling_method: Callable,
    ) -> None:
        super().__init__(
            raw_data=raw_data,
            number_of_sampling_for_centroids=number_of_sampling_for_centroids,
            coreset_size=coreset_size,
            number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
            sampling_method=sampling_method,
            coresets_method="BFL2",
        )

    def _get_coresets_using_selected_approach(
        self, centroids: Union[List[np.ndarray], np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Performs Algorithm 2 from https://arxiv.org/pdf/1612.00889.pdf [BFL2]. This will pick the coreset vectors and its corresponding weights.

        Args:
            centroids (List): The centroids to use for the coreset generation.

        Returns:
            Tuple[List, List]: The coreset vectors and coreset weights.
        """

        number_of_data_points_close_to_a_cluster = {i: 0 for i in range(len(centroids))}
        sum_distance_to_closest_cluster = 0.0
        for data_instance in self._raw_data:
            min_dist, closest_index = super().distance_to_centroids(
                data_instance, centroids
            )
            number_of_data_points_close_to_a_cluster[closest_index] += 1
            sum_distance_to_closest_cluster += min_dist**2

        Prob = np.zeros(len(self._raw_data))
        for i, p in enumerate(self._raw_data):
            min_dist, closest_index = super().distance_to_centroids(p, centroids)
            Prob[i] += min_dist**2 / (2 * sum_distance_to_closest_cluster)
            Prob[i] += 1 / (
                2
                * len(centroids)
                * number_of_data_points_close_to_a_cluster[closest_index]
            )

        if not (0.999 <= sum(Prob) <= 1.001):
            raise ValueError(
                "sum(Prob) = %s; the algorithm should automatically "
                "normalize Prob by construction" % sum(Prob)
            )
        chosen_indices = np.random.choice(
            len(self._raw_data), size=self._coreset_size, p=Prob
        )
        weights = [1 / (self._coreset_size * Prob[i]) for i in chosen_indices]

        return ([self._raw_data[i] for i in chosen_indices], weights)
