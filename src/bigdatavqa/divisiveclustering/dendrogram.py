from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster


class Dendrogram:
    def __init__(self, coreset_data, hierarchial_clustering_sequence) -> None:
        self._coreset_data = self.__create_coreset_data(coreset_data)
        self._hierarchial_clustering_sequence = self.__convert_numbers_to_name(
            hierarchial_clustering_sequence, coreset_data
        )
        self.linkage_matrix = []

    @property
    def coreset_data(self):
        return self._coreset_data

    @coreset_data.setter
    def coreset_data(self, coreset_data):
        self.linkage_matrix = []
        self._coreset_data = coreset_data

    @property
    def hierarchial_clustering_sequence(self):
        return self._hierarchial_clustering_sequence

    @hierarchial_clustering_sequence.setter
    def hierarchial_clustering_sequence(self, hierarchial_clustering_sequence):
        self.linkage_matrix = []
        self._hierarchial_clustering_sequence = hierarchial_clustering_sequence

    def __call__(self) -> List:
        if not self.linkage_matrix:
            self.get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        return self.linkage_matrix

    def __create_coreset_data(self, coreset_data):
        coreset_data.index = coreset_data.Name
        return coreset_data.drop(columns=["Name", "weights"])

    def __convert_numbers_to_name(self, hc_data, coreset_data):
        converted_hc = []
        for hc in hc_data:
            converted_hc.append([coreset_data.Name[num] for num in hc])
        return converted_hc

    def plot_dendrogram(
        self,
        plot_name: str = "dendo.png",
        plot_title: str = "DIANA",
        orientation: str = "top",
        save_plot: bool = False,
    ):
        if not self.linkage_matrix:
            self.get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        fig = plt.figure(figsize=(25, 10), dpi=100)
        plt.title(plot_title)
        dn = dendrogram(
            self.linkage_matrix,
            labels=self._coreset_data.index,
            orientation=orientation,
            color_threshold=1500,
        )
        plt.show()
        if save_plot:
            plt.savefig(plot_name)

    def get_clusters_using_height(self, threshold):
        if not self.linkage_matrix:
            self.get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        return fcluster(self.linkage_matrix, threshold, criterion="distance")

    def get_clusters_using_k(self, k):
        if not self.linkage_matrix:
            self.get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        return fcluster(self.linkage_matrix, k, criterion="maxclust")

    def plot_clusters(self, clusters):
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self._coreset_data.X, self._coreset_data.Y, c=clusters, cmap="prism"
        )
        plt.show()

    def get_linkage_matrix(self, parent):

        if len(parent) < 2:
            index_of_parent = np.argwhere(self._coreset_data.index == parent[0])
            return index_of_parent[0][0]
        children_1, children_2 = self.find_children(
            parent, self._hierarchial_clustering_sequence
        )

        index1 = self.get_linkage_matrix(children_1)
        index2 = self.get_linkage_matrix(children_2)
        self.linkage_matrix.append(
            [
                index1,
                index2,
                self.distance(index1) + self.distance(index2),
                self.cluster_len(index1) + self.cluster_len(index2),
            ]
        )

        return len(self.linkage_matrix) - 1 + 10

    def distance(self, i):
        if i >= len(self._coreset_data):
            return self.linkage_matrix[i - len(self._coreset_data)][2]
        else:
            return sum(self._coreset_data.iloc[i]) / 9

    def cluster_len(self, i):
        if i >= len(self._coreset_data):
            return self.linkage_matrix[i - len(self._coreset_data)][3]
        else:
            return 1

    @staticmethod
    def find_children(parent, hierarchial_clustering_sequence):

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
