from typing import List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster


class Dendrogram:
    def __convert_numbers_to_name(
        self, hierarchical_clustering_sequence: List[int], full_coreset_df: pd.DataFrame
    ) -> List[str]:
        """
        Converts the int in the hierarchial sequence into the instance name. This would be used to plot the leaves of the dendrogram.

        Args:
            hierarchical_clustering_sequence (List[int]): The hierarchical clustering sequence.
            full_coreset_df (pd.DataFrame): The coreset data.

        Returns:
            List[str]: The converted hierarchical clustering sequence.
        """

        converted_hc = []
        for hc in hierarchical_clustering_sequence:
            converted_hc.append([full_coreset_df.Name[num] for num in hc])

        return converted_hc

    def plot_dendrogram(
        self,
        plot_title: Optional[str] = "DIANA",
        orientation: Optional[str] = "top",
        color_threshold: Optional[int] = None,
        colors: Optional[List] = None,
        clusters: Optional[np.ndarray] = None,
    ):
        """
        Plots the dendrogram.

        Args:
            plot_title (str, optional): The plot title. Defaults to "DIANA".
            orientation (str, optional): The orientation of the dendrogram. Defaults to "top".
            color_threshold (int, optional): The color threshold to convert hierarchial clustering into flat clustering. Defaults to None.
            colors (List, optional): The colors for the leaves. Defaults to None.
            clusters (np.ndarray, optional): Flat clustering results from applying threshold. Defaults to None.

        """

        if not self.linkage_matrix:
            self._get_linkage_matrix(self._hierarchical_clustering_sequence[0])

        if clusters is None:
            clusters = np.array([0] * len(self.full_coreset_df))

        fig = plt.figure(figsize=(10, 10), dpi=100)
        plt.title(plot_title)
        dn = dendrogram(
            self.linkage_matrix,
            labels=self.full_coreset_df.index,
            orientation=orientation,
            color_threshold=color_threshold * 100 if colors else None,
        )

        if color_threshold is not None:
            plt.axhline(y=color_threshold, color="r", linestyle="--")

        if colors is not None:
            if len(colors) < len(set(clusters)):
                raise ValueError(
                    "Number of colors should be equal to number of clusters"
                )
            else:
                colors_dict = {
                    self.full_coreset_df.index[i]: colors[j]
                    for i, j in enumerate(clusters)
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
            self._get_linkage_matrix(self._hierarchical_clustering_sequence[0])

        clusters = fcluster(self.linkage_matrix, threshold, criterion="distance")

        self.labels = np.array(clusters) - 1

        return self.labels

    def get_clusters_using_k(self, k: int) -> np.ndarray:
        """
        Get flat clusters from the hierarchical cluster by defining the number of clusters.

        Args:
            k (int): The number of clusters.

        Returns:
            np.ndarray: The flat cluster labels.

        """
        if not self.linkage_matrix:
            self._get_linkage_matrix(self._hierarchical_clustering_sequence[0])

        clusters = fcluster(self.linkage_matrix, k, criterion="maxclust")

        self.clusters = np.array(clusters) - 1

        return self.clusters

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
        full_coreset_df = self.full_coreset_df.copy()
        full_coreset_df["clusters"] = clusters
        for i in range(full_coreset_df.clusters.nunique()):
            data = full_coreset_df[full_coreset_df.clusters == i]
            plt.scatter(data.X, data.Y, c=colors[i], label=f"Cluster {i}")
        if show_annotation:
            for _, row in full_coreset_df.iterrows():
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
            index_of_parent = np.argwhere(self.full_coreset_df.index == parent[0])
            return index_of_parent[0][0]
        children_1, children_2 = self.find_children(
            parent, self.hierarchical_clustering_sequence
        )

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

        return len(self.linkage_matrix) - 1 + len(self.full_coreset_df)

    def _get_cluster_distance(self, i: int) -> float:
        """
        Get the distance between two clusters.

        Args:
            i (int): The index of the cluster.

        Returns:
            float: The distance of the cluster.
        """

        if i >= len(self.full_coreset_df):
            distance = self.linkage_matrix[i - len(self.full_coreset_df[["X", "Y"]])][2]
        else:
            distance = sum(self.full_coreset_df[["X", "Y"]].iloc[i]) / (
                len(self.full_coreset_df) - 1
            )

        return abs(distance)

    def _get_cluster_length(self, i: int):
        """
        Get the length of the cluster.

        Args:
            i (int): The index of the cluster.

        Returns:
            int: The length of the cluster.
        """

        if i >= len(self.full_coreset_df):
            return self.linkage_matrix[i - len(self.full_coreset_df)][3]
        else:
            return 1

    @staticmethod
    def find_children(
        parent: List[Union[str, int]],
        hierarchical_clustering_sequence: List[Union[str, int]],
    ) -> List:
        """
        Find the children of a given parent cluster.

        Args:
            parent (List): The parent cluster.
            hierarchical_clustering_sequence (List): The hierarchical clustering sequence.

        Returns:
            List: The children of the parent cluster.
        """

        parent_position = hierarchical_clustering_sequence.index(parent)

        found = 0
        children = []
        for i in range(parent_position + 1, len(hierarchical_clustering_sequence)):
            if any(item in hierarchical_clustering_sequence[i] for item in parent):
                children.append(hierarchical_clustering_sequence[i])
                found += 1
                if found == 2:
                    break

        return children

    def plot_hierarchial_split(self):
        """
        Plots the flat clusters at each iteration of the hierarchical clustering.

        Args:
            hierarchical_clustering_sequence (List): The hierarchical clustering sequence.
            full_coreset_df (pd.DataFrame): The full coreset data.
        """
        parent_clusters = [
            parent_cluster
            for parent_cluster in self.hierarchical_clustering_sequence
            if len(parent_cluster) > 1
        ]
        x_grid = int(np.sqrt(len(parent_clusters)))
        y_grid = int(np.ceil(len(parent_clusters) / x_grid))

        fig, axs = plt.subplots(x_grid, y_grid, figsize=(12, 12))

        for i, parent_cluster in enumerate(parent_clusters):
            parent_position = self.hierarchical_clustering_sequence.index(
                parent_cluster
            )
            children = Dendrogram.find_children(
                parent_cluster, self.hierarchical_clustering_sequence
            )
            coreset_for_parent_cluster = self.full_coreset_df.loc[parent_cluster]
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
