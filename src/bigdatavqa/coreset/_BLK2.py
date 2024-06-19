from typing import Callable, List, Tuple, Union

from ._coreset import Coreset

import numpy as np


class BLK2(Coreset):
    def __init__(
        self,
        raw_data: np.ndarray,
        number_of_sampling_for_centroids: int,
        coreset_size: int,
        number_of_coresets_to_evaluate: int,
        sampling_method: Callable,
        k_value_for_BLK2: int,
    ) -> None:
        super().__init__(
            raw_data=raw_data,
            number_of_sampling_for_centroids=number_of_sampling_for_centroids,
            coreset_size=coreset_size,
            number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
            sampling_method=sampling_method,
            coresets_method="BFL2",
        )
        self._k_value_for_BLK2 = k_value_for_BLK2

    @property
    def k_value_for_BLK2(self) -> int:
        return self._k_value_for_BLK2

    @k_value_for_BLK2.setter
    def k_value_for_BLK2(self, k_value_for_BLK2: int) -> None:
        self._k_value_for_BLK2 = k_value_for_BLK2

    def _get_coresets_using_selected_approach(
        self,
        centroids: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Performs Algorithm 2 from  https://arxiv.org/pdf/1703.06476.pdf.

        Args:
            centroids (List[np.ndarray]): The centroids to use for the coreset generation.

        Returns:
            Tuple[List, List]: The coreset vectors and coreset weights.
        """

        alpha = 16 * (np.log2(self._k_value_for_BLK2) + 2)

        B_i_totals = [0] * len(centroids)
        B_i = [np.empty_like(self._raw_data) for _ in range(len(centroids))]
        for data_instance in self._raw_data:
            _, closest_index = self.distance_to_centroids(data_instance, centroids)
            B_i[closest_index][B_i_totals[closest_index]] = data_instance
            B_i_totals[closest_index] += 1

        c_phi = sum(
            [
                self.distance_to_centroids(data_instance, centroids)[0] ** 2
                for data_instance in self._raw_data
            ]
        ) / len(self._raw_data)

        p = np.zeros(len(self._raw_data))

        sum_dist = {i: 0.0 for i in range(len(centroids))}
        for i, data_instance in enumerate(self._raw_data):
            dist, closest_index = self.distance_to_centroids(data_instance, centroids)
            sum_dist[closest_index] += dist**2

        for i, data_instance in enumerate(self._raw_data):
            p[i] = (
                2
                * alpha
                * self.distance_to_centroids(data_instance, centroids)[0] ** 2
                / c_phi
            )

            closest_index = self.distance_to_centroids(data_instance, centroids)[1]
            p[i] += (
                4
                * alpha
                * sum_dist[closest_index]
                / (B_i_totals[closest_index] * c_phi)
            )

            p[i] += 4 * len(self._raw_data) / B_i_totals[closest_index]
        p = p / sum(p)

        chosen_indices = np.random.choice(
            len(self._raw_data), size=self._coreset_size, p=p
        )
        weights = [1 / (self._coreset_size * p[i]) for i in chosen_indices]

        return [self._raw_data[i] for i in chosen_indices], weights
