from typing import Callable, List, Tuple, Union

from ._coreset import Coreset

import numpy as np

from pydantic import BaseModel, Field
from ._coreset import CoresetConfig


class BLK2Config(BaseModel):
    """
    Configuration class for BLK2.

    Args:
        k_value_for_BLK2 (int): The k value for BLK2.
    """

    k_value_for_BLK2: int = Field(..., title="The k value for BLK2.")


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
        """
        Initialize the BLK2 object.

        Args:
            raw_data (np.ndarray): The raw data used for coreset construction.
            number_of_sampling_for_centroids (int): The number of sampling iterations for selecting centroids.
            coreset_size (int): The desired size of the coreset.
            number_of_coresets_to_evaluate (int): The number of coreset candidates to evaluate.
            sampling_method (Callable): The method used for sampling data points.
            k_value_for_BLK2 (int): The value of k used in the BLK2 algorithm.

        Returns:
            None
        """
        base_coreset_config = CoresetConfig(
            number_of_sampling_for_centroids=number_of_sampling_for_centroids,
            coreset_size=coreset_size,
            number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
            sampling_method=sampling_method,
        )
        super().__init__(
            raw_data=raw_data,
            coreset_config=base_coreset_config,
        )
        self.BLK2_config = BLK2Config(k_value_for_BLK2=k_value_for_BLK2)

    @property
    def k_value_for_BLK2(self) -> int:
        return self.BLK2_config.k_value_for_BLK2

    @k_value_for_BLK2.setter
    def k_value_for_BLK2(self, k_value_for_BLK2: int) -> None:
        self.BLK2_config.k_value_for_BLK2 = k_value_for_BLK2

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

        alpha = 16 * (np.log2(self.BLK2_config.k_value_for_BLK2) + 2)

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
            len(self._raw_data), size=self.coreset_config.coreset_size, p=p
        )
        weights = [
            1 / (self.coreset_config.coreset_size * p[i]) for i in chosen_indices
        ]

        return [self._raw_data[i] for i in chosen_indices], weights
