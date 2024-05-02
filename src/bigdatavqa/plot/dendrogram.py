from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster


class Dendrogram:
    def __init__(
        self, coreset_data: pd.DataFrame, hierarchial_clustering_sequence: List[Union[str, int]]
    ) -> None:
        self._coreset_data = self.__create_coreset_data(coreset_data)
        self._hierarchial_clustering_sequence = self.__convert_numbers_to_name(
            hierarchial_clustering_sequence, coreset_data
        )
        self.linkage_matrix = []

    @property
    def coreset_data(self) -> pd.DataFrame:
        return self._coreset_data

    @coreset_data.setter
    def coreset_data(self, coreset_data: pd.DataFrame) -> None:
        self.linkage_matrix = []
        self._coreset_data = coreset_data

    @property
    def hierarchial_clustering_sequence(self) -> List[Union[str, int]]:
        return self._hierarchial_clustering_sequence

    @hierarchial_clustering_sequence.setter
    def hierarchial_clustering_sequence(
        self, hierarchial_clustering_sequence: List[Union[str, int]]
    ) -> None:
        self.linkage_matrix = []
        self._hierarchial_clustering_sequence = hierarchial_clustering_sequence

    def __call__(self) -> List:
        if not self.linkage_matrix:
            self._get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        return self.linkage_matrix

    def __create_coreset_data(self, coreset_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates coreset data that can be used for plotting.

        Args:
            coreset_data (pd.DataFrame): The coreset data.

        Returns:
            pd.DataFrame: The coreset data.
        """

        _coreset_data = coreset_data.copy()
        _coreset_data.index = _coreset_data.Name

        return _coreset_data.drop(columns=["Name", "weights"])

    def __convert_numbers_to_name(
        self, hierarchial_clustering_sequence: List[int], coreset_data: pd.DataFrame
    ) -> List[str]:
        """
        Converts the int in the hierarchial sequence into the instance name. This would be used to plot the leaves of the dendrogram.

        Args:
            hierarchial_clustering_sequence (List[int]): The hierarchical clustering sequence.
            coreset_data (pd.DataFrame): The coreset data.

        Returns:
            List[str]: The converted hierarchical clustering sequence.
        """

        converted_hc = []
        for hc in hierarchial_clustering_sequence:
            converted_hc.append([coreset_data.Name[num] for num in hc])

        return converted_hc

    def plot_dendrogram(
        self,
        plot_title: Optional[str] = "DIANA",
        orientation: Optional[str] = "top",
        color_threshold: Optional[int] = None,
        colors: Optional[List] = None,
        clusters: Optional[np.ndarray] = None,
        link_color_func: Optional[Callable] = None,
    ):
        """
        Plots the dendrogram.

        Args:
            plot_title (str, optional): The plot title. Defaults to "DIANA".
            orientation (str, optional): The orientation of the dendrogram. Defaults to "top".
            color_threshold (int, optional): The color threshold to convert hierarchial clustering into flat clustering. Defaults to None.
            colors (List, optional): The colors for the leaves. Defaults to None.
            clusters (np.ndarray, optional): Flat clustering results from applying threshold. Defaults to None.
            link_color_func (Callable, optional): Function to colour the branches. Defaults to None.
        """

        if not self.linkage_matrix:
            self._get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        if clusters is None:
            clusters = np.array([0] * len(self._coreset_data))

        fig = plt.figure(figsize=(10, 10), dpi=100)
        plt.title(plot_title)
        dn = dendrogram(
            self.linkage_matrix,
            labels=self._coreset_data.index,
            orientation=orientation,
            color_threshold=color_threshold * 100 if colors else None,
        )

        if color_threshold is not None:
            plt.axhline(y=color_threshold, color="r", linestyle="--")

        if colors is not None:
            if len(colors) < len(set(clusters)):
                raise ValueError("Number of colors should be equal to number of clusters")
            else:
                colors_dict = {
                    self._coreset_data.index[i]: colors[j] for i, j in enumerate(clusters)
                }

                ax = plt.gca()
                xlbls = ax.get_xmajorticklabels()
                for lbl in xlbls:
                    lbl.set_color(colors_dict[lbl.get_text()])

        plt.show()

    def get_clusters_using_height(self, threshold: float) -> np.ndarray:
        """
        Get flat clusters from the hierarchical clustering using a threshold.

        Args:
            threshold (float): The height threshold to convert.

        Returns:
            np.ndarray: The flat cluster labels.
        """

        if not self.linkage_matrix:
            self._get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        clusters = fcluster(self.linkage_matrix, threshold, criterion="distance")

        return np.array(clusters) - 1

    def get_clusters_using_k(self, k: int) -> np.ndarray:
        """
        Get flat clusters from the hierarchical cluster by defining the number of clusters.

        Args:
            k (int): The number of clusters.

        Returns:
            np.ndarray: The flat cluster labels.

        """
        if not self.linkage_matrix:
            self._get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        clusters = fcluster(self.linkage_matrix, k, criterion="maxclust")

        return np.array(clusters) - 1

    def plot_clusters(
        self,
        clusters: np.ndarray,
        colors: List[str],
        plot_title: str,
        show_annotation: Optional[bool] = False,
    ):
        """
        Plot the flat clusters.

        Args:
            clusters (np.ndarray): The flat clusters.
            colors (List[str]): The colors for the clusters.
            plot_title (str): The plot title.
            show_annotation (bool, optional): Show annotation. Defaults to False.

        """
        if len(colors) < len(set(clusters)):
            raise ValueError("Number of colors should be equal to number of clusters")
        coreset_data = self._coreset_data.copy()
        coreset_data["clusters"] = clusters
        for i in range(coreset_data.clusters.nunique()):
            data = coreset_data[coreset_data.clusters == i]
            plt.scatter(data.X, data.Y, c=colors[i], label=f"Cluster {i}")
        if show_annotation:
            for _, row in coreset_data.iterrows():
                plt.annotate(row.name, (row.X, row.Y))
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(plot_title)
        plt.show()

    def _get_linkage_matrix(self, parent: List[str]) -> int:
        """
        Create the linkage matrix for the dendrogram and returns the index of the new branch.

        Args:
            parent (List[str]): The parent cluster.

        Returns:
            List: The linkage matrix.
        """

        if len(parent) < 2:
            index_of_parent = np.argwhere(self._coreset_data.index == parent[0])
            return index_of_parent[0][0]
        children_1, children_2 = self.find_children(parent, self._hierarchial_clustering_sequence)

        index1 = self._get_linkage_matrix(children_1)
        index2 = self._get_linkage_matrix(children_2)
        self.linkage_matrix.append(
            [
                index1,
                index2,
                self._get_cluster_distance(index1) + self._get_cluster_distance(index2),
                self._get_cluster_length(index1) + self._get_cluster_length(index2),
            ]
        )

        return len(self.linkage_matrix) - 1 + len(self.coreset_data)

    def _get_cluster_distance(self, i: int) -> float:
        """
        Get the distance between two clusters.

        Args:
            i (int): The index of the cluster.

        Returns:
            float: The distance of the cluster.
        """

        if i >= len(self._coreset_data):
            distance = self.linkage_matrix[i - len(self._coreset_data)][2]
        else:
            distance = sum(self._coreset_data.iloc[i]) / (len(self.coreset_data) - 1)

        return abs(distance)

    def _get_cluster_length(self, i: int):
        """
        Get the length of the cluster.

        Args:
            i (int): The index of the cluster.

        Returns:
            int: The length of the cluster.
        """

        if i >= len(self._coreset_data):
            return self.linkage_matrix[i - len(self._coreset_data)][3]
        else:
            return 1

    @staticmethod
    def find_children(
        parent: List[Union[str, int]], hierarchial_clustering_sequence: List[Union[str, int]]
    ) -> List:
        """
        Find the children of a given parent cluster.

        Args:
            parent (List): The parent cluster.
            hierarchial_clustering_sequence (List): The hierarchical clustering sequence.

        Returns:
            List: The children of the parent cluster.
        """

        parent_position = hierarchial_clustering_sequence.index(parent)

        found = 0
        children = []
        for i in range(parent_position + 1, len(hierarchial_clustering_sequence)):
            if any(item in hierarchial_clustering_sequence[i] for item in parent):
                children.append(hierarchial_clustering_sequence[i])
                found += 1
                if found == 2:
                    break

        return children

    @staticmethod
    def plot_hierarchial_split(
        hierarchial_clustering_sequence: List[Union[str, int]], full_coreset_df: pd.DataFrame
    ):
        """
        Plots the flat clusters at each iteration of the hierarchical clustering.

        Args:
            hierarchial_clustering_sequence (List): The hierarchical clustering sequence.
            full_coreset_df (pd.DataFrame): The full coreset data.
        """

        parent_clusters = [
            parent_cluster
            for parent_cluster in hierarchial_clustering_sequence
            if len(parent_cluster) > 1
        ]
        x_grid = int(np.sqrt(len(parent_clusters)))
        y_grid = int(np.ceil(len(parent_clusters) / x_grid))

        fig, axs = plt.subplots(x_grid, y_grid, figsize=(12, 12))

        for i, parent_cluster in enumerate(parent_clusters):
            parent_position = hierarchial_clustering_sequence.index(parent_cluster)
            children = Dendrogram.find_children(parent_cluster, hierarchial_clustering_sequence)
            coreset_for_parent_cluster = full_coreset_df.loc[parent_cluster]
            coreset_for_parent_cluster["cluster"] = 1
            coreset_for_parent_cluster.loc[children[0], "cluster"] = 0

            ax = axs[i // 3, i % 3]
            ax.scatter(
                coreset_for_parent_cluster["X"],
                coreset_for_parent_cluster["Y"],
                c=coreset_for_parent_cluster["cluster"],
            )
            for _, row in coreset_for_parent_cluster.iterrows():
                ax.annotate(row["Name"], (row["X"], row["Y"]))

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Clustering at iteration {parent_position}")

        plt.tight_layout()
        plt.show()
