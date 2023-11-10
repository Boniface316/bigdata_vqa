import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")


class Dendrogram:
    def __init__(
        self,
        hierarchial_clustering_sequence,
        coreset_data,
        raw_data,
        dendrogram_file="/workspace/vqa/data/results/divisive_clustering/dendrogram_df",
        use_normalized_coreset=True,
        cluster_reference_dict=None,
        centroids_dist_df=None,
        dendrogram_df=None,
    ) -> None:
        self.hierarchial_clustering_sequence = hierarchial_clustering_sequence
        self.raw_data = raw_data
        self.use_normalized_coreset = use_normalized_coreset
        self.cluster_reference_dict = cluster_reference_dict
        self.centroids_dist_df = centroids_dist_df
        self.dendrogram_df = dendrogram_df
        self.dendrogram_file = dendrogram_file
        if use_normalized_coreset:
            coreset_data = coreset_data[["X_norm", "Y_norm", "weights_norm", "name"]]
            coreset_data.columns = ["X", "Y", "weights", "name"]
            self.coreset_data = coreset_data
        else:
            self.coreset_data = coreset_data[["X", "Y", "weights", "name"]]

    def get_centroid_dist_df(self):
        centroids = self.get_centroid_coords(
            self.hierarchial_clustering_sequence, self.coreset_data
        )
        centroids_df = self.get_centroid_dist(centroids)
        centdoids_distance_df = self.rename_cols_rows(centroids_df, self.coreset_data)
        if self.centroids_dist_df is None:
            self.centroids_dist_df = centdoids_distance_df

        return centdoids_distance_df

    def get_centroid_coords(
        self, hierarchial_clustering_sequence=None, coreset_data=None
    ):
        if hierarchial_clustering_sequence is None:
            hierarchial_clustering_sequence = self.hierarchial_clustering_sequence
        if coreset_data is None:
            coreset_data = self.coreset_data
        centroids = {}
        coreset_data = coreset_data[["X", "Y"]]

        for posn in hierarchial_clustering_sequence:
            if isinstance(posn[0], str):
                posn = [ord(c) - ord("A") for c in posn]
            coreset_data_selected = coreset_data.iloc[posn]

            centroid = np.mean(coreset_data_selected.to_numpy(), axis=0)

            centroids.update({str(posn): centroid})

        return centroids

    def get_centroid_dist(self, centroids):
        centroid_df = pd.DataFrame(
            0, index=np.arange(len(centroids)), columns=centroids.keys()
        )
        centroid_df.index = centroids.keys()
        for idx in centroid_df.index:
            for col in centroid_df.columns:
                a = centroids[idx]
                b = centroids[col]
                centroid_df.loc[idx, col] = np.linalg.norm(a - b)
        return centroid_df

    def rename_cols_rows(self, centroids_df, coreset_data):
        cluster_names = {}

        for idx in coreset_data.index:
            cluster_names.update({str(idx): coreset_data.name[idx]})

        col = centroids_df.columns[0]

        col_names = []

        for col in centroids_df.columns:
            a = re.sub(r"[\[\]]", "", col)
            b = ""
            for chr_val in a.split(","):
                chr_val = chr_val.replace(" ", "")
                b += cluster_names[str(chr_val)]
            col_names.append(b)

        centroids_df.columns = col_names
        centroids_df.index = col_names

        return centroids_df

    def get_cluster_reference_dict(self, centroids_dist_df=None):
        if centroids_dist_df is None:
            centroids_dist_df = self.centroids_dist_df

        cluster_reference_dict = {}
        colnames = centroids_dist_df.columns

        for i, cluster in enumerate(self.hierarchial_clustering_sequence):
            clus_name = colnames[i]
            cluster_reference_dict.update({str(cluster): clus_name})

        if self.cluster_reference_dict is None:
            self.cluster_reference_dict = cluster_reference_dict

        return cluster_reference_dict

    def create_empty_dendo_df(self):
        return pd.DataFrame(
            columns=[
                "X1",
                "Y1",
                "X2",
                "Y2",
                "VX",
                "VY",
                "Cluster1",
                "Cluster2",
                "Parent",
                "Singleton",
            ],
            index=range(len(self.hierarchial_clustering_sequence)),
        )

    def get_cluster_position(self, current_position, dendrogram_df=None):
        if dendrogram_df is None:
            dendrogram_df = self.dendrogram_df
        current_cluster = self.cluster_reference_dict[
            str(self.hierarchial_clustering_sequence[current_position])
        ]

        return self.find_parent_cluster_posn(current_cluster, dendrogram_df)

    def find_parent_cluster_posn(self, parent_cluster, dendrogram_df):
        clus1 = dendrogram_df[dendrogram_df.Cluster1 == parent_cluster]
        clus2 = dendrogram_df[dendrogram_df.Cluster2 == parent_cluster]

        cluster_position = None

        if len(clus1) > 0:
            cluster_position = "C1"
        elif len(clus2) > 0:
            cluster_position = "C2"
        else:
            "cant find"

        return cluster_position

    def get_parent_location(self, current_posn):
        hc = self.hierarchial_clustering_sequence
        parent_hc = hc[:current_posn]
        parent_hc = parent_hc[::-1]

        child_value = hc[current_posn]
        return self.find_parent_location(parent_hc, child_value)

    def find_parent_location(self, parent_hc, child_value):
        for i, hc_val in enumerate(parent_hc):
            for idx in hc_val:
                if idx == child_value[0]:
                    parent_loc = len(parent_hc) - 1 - i
                    return parent_loc

    def get_xy_values(
        self,
        parent_location: int,
        cluster_position: int,
        dendrogram_df=None,
    ):
        if dendrogram_df is None:
            dendrogram_df = self.dendrogram_df
        if cluster_position == "C1":
            x_start = dendrogram_df.iloc[parent_location]["X1"][1]
            y_val = dendrogram_df.iloc[parent_location]["Y1"][1]
        else:
            x_start = dendrogram_df.iloc[parent_location]["X2"][1]
            y_val = dendrogram_df.iloc[parent_location]["Y2"][1]

        return x_start, y_val

    def find_children(self, parent_position):
        index_vals = self.hierarchial_clustering_sequence[parent_position]

        if len(index_vals) == 1:
            return []

        child_posisition = []
        child_list = []
        children_list = []

        for instance in index_vals:
            for i in range(
                parent_position + 1, len(self.hierarchial_clustering_sequence)
            ):
                idx_val = self.hierarchial_clustering_sequence[i]

                if instance in idx_val:
                    if instance not in child_list:
                        child_posisition.append(i)
                        break

        children_position = list(dict.fromkeys(child_posisition))

        if self.cluster_reference_dict is None:
            for child in children_position:
                child_list.append(self.hierarchial_clustering_sequence[child])

            return child_list

        else:
            for child in children_position:
                children_list.append(self.hierarchial_clustering_sequence[child])

            if children_list:
                child1_str = str(children_list[0])
                child2_str = str(children_list[1])

                child1_name = self.cluster_reference_dict[child1_str]
                child2_name = self.cluster_reference_dict[child2_str]

                return [child1_name, child2_name]
            else:
                return children_list

    def get_row_values_for_dendrogram(
        self, parent_position, x_start, y_start, buffer_1, buffer_2
    ):
        children_lst = self.find_children(parent_position)
        parent_cluster = self.cluster_reference_dict[
            str(self.hierarchial_clustering_sequence[parent_position])
        ]
        if children_lst:
            child1_name, child2_name = children_lst

        else:
            return [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                None,
                None,
                parent_cluster,
                True,
            ]

        a = self.centroids_dist_df[[parent_cluster]].loc[child1_name]
        a = float(a)

        b = self.centroids_dist_df[[parent_cluster]].loc[child2_name]
        b = float(b)

        height = [y_start + buffer_1, y_start + buffer_2]

        x1, y1 = [x_start, x_start + a], [height[0], height[0]]
        x2, y2 = [x_start, x_start + b], [height[1], height[1]]
        vx, vy = [x_start, x_start], [height[0], height[1]]

        return [x1, y1, x2, y2, vx, vy, child1_name, child2_name, parent_cluster, False]

    def find_longest_x(self, dendrogram_df):
        x1_list = []

        for a in dendrogram_df.X1:
            x1_list.append(a[1])

        maxX1 = max(x1_list)

        x2_list = []

        for a in dendrogram_df.X2:
            x2_list.append(a[1])

        maxX2 = max(x2_list)

        longest_x = max(maxX1, maxX2)

        return longest_x

    def extend_singletons(
        self,
        longest_x,
        dendrogram_df=None,
    ):
        if dendrogram_df is None:
            dendrogram_df = self.dendrogram_df

        singleton_positions = dendrogram_df.index[dendrogram_df.Singleton]

        for current_position in singleton_positions:
            parent_cluster = self.cluster_reference_dict[
                str(self.hierarchial_clustering_sequence[current_position])
            ]

            cluster1 = dendrogram_df[dendrogram_df.Cluster1 == parent_cluster]
            cluster2 = dendrogram_df[dendrogram_df.Cluster2 == parent_cluster]

            if len(cluster1) > 0:
                cluster_position = "C1"
            elif len(cluster2) > 0:
                cluster_position = "C2"
            else:
                ValueError("Unable to find cluster position")

            parent_location = self.get_parent_location(current_position)

            xy_coords = dendrogram_df.loc[parent_location]

            if cluster_position == "C1":
                x1 = [xy_coords[0][1], longest_x]
                y1 = [xy_coords[1][1], xy_coords[1][1]]
            elif cluster_position == "C2":
                x1 = [xy_coords[2][1], longest_x]
                y1 = [xy_coords[3][1], xy_coords[3][1]]
            else:
                ValueError("Unable to find cluster position")

            dendrogram_df.loc[current_position]["X1"] = x1
            dendrogram_df.loc[current_position]["Y1"] = y1

        return dendrogram_df

    def plot_dendrogram(
        self,
        current_position=None,
        dendrogram_df=None,
        plot_name=None,
        vertical_line=None,
        save_image=True,
    ):
        if current_position is not None:
            plot_name = "dendrogram_df_plot_draft.png"
            if dendrogram_df is None:
                dendrogram_df = self.dendrogram_df
            dendrogram_df = dendrogram_df.head(current_position + 1)
            figsize = (20, 10)

        else:
            if dendrogram_df is None:
                ValueError("dendrogram_df is not provided")
            else:
                figsize = (5, 10)

        longest_x = self.find_longest_x(dendrogram_df)
        dendrogram_df = self.extend_singletons(longest_x, dendrogram_df)
        fig, ax = plt.subplots(figsize=figsize)

        for idx, plot_data in dendrogram_df.iterrows():
            x1, y1, x2, y2, vx, vy, C1, C2, Parent, Singleton = plot_data
            if Singleton:
                plt.plot(x1, y1, color="black", marker=" ")
                ax.annotate(Parent, (longest_x + 0.2, y1[0]), fontsize=10)
            else:
                if idx == len(dendrogram_df) - 1:
                    posn_1 = (x1[1], y1[1])
                    posn_2 = (x2[1], y2[1])
                    plt.annotate(C1, xy=posn_1, color="red")
                    plt.annotate(C2, xy=posn_2, color="green")

                plt.plot(x1, y1, x2, y2, vx, vy, color="black", marker=" ")
        ax.set_yticklabels([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if vertical_line is not None:
            ax.axvline(x=vertical_line, color="r")

        if save_image:
            plt.savefig(plot_name, bbox_inches="tight")

    def plot_dendrogram_manually(self, current_position, save_image=True):
        if current_position == 0:
            self.dendrogram_df = self.create_empty_dendo_df()
            self.get_centroid_dist_df()
            self.get_cluster_reference_dict(self.centroids_dist_df)
            x_start = 0
            y_start = 0
        else:
            if self.centroids_dist_df is None:
                self.get_centroid_dist_df()
            if self.cluster_reference_dict is None:
                self.get_cluster_reference_dict(self.centroids_dist_df)
            self.dendrogram_df = pd.read_pickle(f"{self.dendrogram_file}.pkl")
            cluster_position = self.get_cluster_position(current_position)
            parent_location = self.get_parent_location(current_position)
            x_start, y_start = self.get_xy_values(parent_location, cluster_position)

        buffers = input("Enter buffer 1 and buffer 2: ")
        buffer_1, buffer_2 = buffers.split(" ")
        buffer_1 = int(buffer_1)
        buffer_2 = int(buffer_2)

        dendrogram_row_calues = self.get_row_values_for_dendrogram(
            current_position,
            x_start,
            y_start,
            buffer_1,
            buffer_2,
        )

        self.dendrogram_df.iloc[current_position] = dendrogram_row_calues

        self.plot_dendrogram(current_position, save_image=save_image)

        satisfied = input("Are you satisfied with the plot? (y/n): ")

        if satisfied == "y":
            pd.to_pickle(self.dendrogram_df, f"{self.dendrogram_file}.pkl")
