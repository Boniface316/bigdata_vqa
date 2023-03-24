import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


class DataUtils:
    def __init__(self, data_folder: str = None):
        if data_folder is None:
            self.data_folder = os.getcwd()
        else:
            self.data_folder = data_folder

    def create_new_data_set(
        self,
        n_samples: int,
        centers: int,
        n_features: int,
        random_state: int = 10,
        save_file: bool = True,
    ):
        """
        Create a new data set

        Args:
            n_samples (int): Number of data samples
            centers (int): centers of the data set
            n_features (int): number of dimension of the plot
            random_state (int, optional): random state for reproduction. Defaults to 10.
            save_file (bool, optional): save file or not. Defaults to True.

        Returns:
            created data vector
        """

        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            n_features=n_features,
            random_state=random_state,
        )
        data_vec_pd = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1]))
        data_vectors = data_vec_pd.to_numpy()
        if save_file:
            file_name = (
                self.data_folder
                + "/data/centers_"
                + str(centers)
                + "_original_data.npy"
            )
            np.save(file_name, data_vectors)
        return data_vectors

    def get_raw_data(self, centers: int):
        """
        Get the saved raw data

        Args:
            centers (int): number of centers

        Returns:
            data vectors
        """
        original_data_file = (
            self.data_folder
            + "/data/"
            + "centers_"
            + str(centers)
            + "_original_data.npy"
        )
        data_vectors = np.load(original_data_file)
        return data_vectors

    def save_coresets(
        self,
        best_coreset_vectors: np.ndarray,
        best_coreset_weights: np.ndarray,
        coreset_numbers: int,
        centers: int,
    ) -> None:
        """
            Save coresets

        Args:
            best_coreset_vectors: Coreset vectors that we want to save
            best_coreset_weights: Coreset weights that we want to save
            coreset_numbers: Number of coresets
            centers: number of blobs
        """

        coreset_vector_file = (
            self.data_folder
            + "/data/"
            + str(coreset_numbers)
            + "_coreset_"
            + str(centers)
            + "centers_vectors.npy"
        )
        coreset_weight_file = (
            self.data_folder
            + "/data/"
            + str(coreset_numbers)
            + "_coreset_"
            + str(centers)
            + "centers_weights.npy"
        )

        np.save(coreset_vector_file, best_coreset_vectors)
        np.save(coreset_weight_file, best_coreset_weights)

    def get_files(self, coreset_numbers: int, centers: int):
        """
        Get coreset vectors and weights and original data

        Args:
            coreset_numbers: number of coresets
            centers: number of blobs

        Returns:
            coreset cectors, weights and orignal data
        """

        original_data_file = (
            self.data_folder
            + "/data/"
            + "centers_"
            + str(centers)
            + "_original_data.npy"
        )
        coreset_vector_file = (
            self.data_folder
            + "/data/"
            + str(coreset_numbers)
            + "_coreset_"
            + str(centers)
            + "centers_vectors.npy"
        )
        coreset_weights_file = (
            self.data_folder
            + "/data/"
            + str(coreset_numbers)
            + "_coreset_"
            + str(centers)
            + "centers_weights.npy"
        )
        coreset_vectors = np.load(coreset_vector_file)
        coreset_weights = np.load(coreset_weights_file)
        data_vec = np.load(original_data_file)

        return coreset_vectors, coreset_weights, data_vec

    def save_object(
        self,
        type: str,
        coreset_numbers: int,
        centers: int,
        depth: int,
        hc: list,
        df: pd.DataFrame,
        str_prob_dict: Dict,
    ):

        """_summary_

        Args:
            type: type of algorithm used for clustering
            coreset_numbers : Number of coresets
            centers: Number of blobs
            depth: depth of circuit
            hc: list of hierarchy pattern
            df: dataframe of the coresets
            str_prob_dict: Dictionary with probabilities
        """
        file_name_hc = (
            self.data_folder
            + "/data/"
            + type
            + "/"
            + type
            + "_hc_coreset_"
            + str(coreset_numbers)
            + "_centers_"
            + str(centers)
            + "_p_"
            + str(depth)
            + ".pickle"
        )
        open_file = open(file_name_hc, "wb")
        pickle.dump(hc, open_file)
        open_file.close()

        file_name_pd = (
            self.data_folder
            + "/data/"
            + type
            + "/"
            + type
            + "_pd_coreset_"
            + str(coreset_numbers)
            + "_centers_"
            + str(centers)
            + "_p_"
            + str(depth)
            + ".pickle"
        )
        open_file = open(file_name_pd, "wb")
        pickle.dump(df, open_file)
        open_file.close()

        file_name_dict = (
            self.data_folder
            + "/data/"
            + type
            + "/"
            + type
            + "_probs_dict_coreset_"
            + str(coreset_numbers)
            + "_centers_"
            + str(centers)
            + "_p_"
            + str(depth)
            + ".pickle"
        )
        open_file = open(file_name_dict, "wb")
        pickle.dump(str_prob_dict, open_file)
        open_file.close()

    def load_object(
        self,
        output_type: str,
        coreset_numbers: int,
        centers: int,
        depth: int,
        obj_type: str,
    ):
        """
        Load the object

        Args:
            output_type: Output generated by i.e "QAOA" or "VQE"
            coreset_numbers: number of coreset
            centers: number of blobs
            depth: depth of circuit
            obj_type: object of interest


        Returns:
           Object
        """

        obj_types = ["hc", "pd", "probs_dict"]

        if obj_type not in obj_types:
            raise RuntimeError("Accepted obj_types are {}".format(obj_types))
        else:
            obj_type = "_" + obj_type + "_coreset_"

            file_name_hc = (
                self.data_folder
                + "/data/"
                + output_type
                + "/"
                + output_type
                + obj_type
                + str(coreset_numbers)
                + "_centers_"
                + str(centers)
                + "_p_"
                + str(depth)
                + ".pickle"
            )
            # print(file_name_hc)
            open_file = open(file_name_hc, "rb")
            loaded_list = pickle.load(open_file)
            open_file.close()

        return loaded_list

    def get_plot_name(
        self,
        type_of_output: str,
        coreset_numbers: int,
        centers: int,
        depth: int,
        parent_posn: int,
    ):

        """
        Get the name of th plot

        Args:
            type_of_output: algorithm that generated the output
            coreset_number: number of coresets
            centers: number of blobs
            depth: depth of a circuit
            parent_posn: position of the parent cluster

        Returns:
            plot name
        """
        plot_name = (
            self.data_folder
            + "/plots/"
            + type_of_output
            + "/coresets_"
            + str(coreset_numbers)
            + "_centers_"
            + str(centers)
            + "_p_"
            + str(depth)
            + "_split_"
            + str(parent_posn)
            + ".png"
        )
        return plot_name

    def save_dendo_data(self, dendo_data: pd.DataFrame, type_of_plot: str):
        """
        Save the dendogram related data

        Args:
            dendo_data: Dataframe of dendrogram data
            type_of_plot: Algorithm that created the data
        """
        file_name = (
            self.data_folder
            + "data/"
            + type_of_plot
            + "/"
            + type_of_plot
            + "_dendo_data_for_plot.pickle"
        )
        open_file = open(file_name, "wb")
        pickle.dump(dendo_data, open_file)

    def load_cluster_colors(self, type_of_plot: str):
        """
        Load the colors of clusters

        Args:
            type_of_plot: algorithm that generated the output

        Returns:
            loaded file
        """
        file_name = (
            self.data_folder
            + "/data/"
            + type_of_plot
            + "/"
            + type_of_plot
            + "_color_dict.pickle"
        )
        open_file = open(file_name, "rb")
        color_dict = pickle.load(open_file)
        open_file.close()

        return color_dict

    def write_i(self, i: int, single_clusters: int):
        """
        Text file created to keep track of iterations

        Args:
            i: iteration step
            single_clusters: number of singleton clusters
        """
        f = open(self.data_folder + "posn.txt", "w+")
        f.write(str(i) + "\r\n")
        f.write(str(single_clusters))
        f.close()

    def save_raw_object(
        self, type_of_output, coreset_numbers, centers, depth, df, i, special_char=""
    ):
        """
        Save raw object as a pickle

        Args:
            type_of_output: algorithm that created the outcome
            coreset_numbers: number of coresets
            centers: number of blobs
            depth: depth of a circuit
            df: data frame of the data
            i: hierachy iteration
            special_char (str, optional): Any special charcters that needed to be added to the file. Defaults to "".
        """
        file_name = (
            self.data_folder
            + "/data/"
            + type_of_output
            + "/raw/"
            + type_of_output
            + "_df_coreset_"
            + str(coreset_numbers)
            + "_centers_"
            + str(centers)
            + "_p_"
            + str(depth)
            + "_i_"
            + str(i)
            + special_char
            + ".pickle"
        )
        open_file = open(file_name, "wb")
        pickle.dump(df, open_file)

    def load_raw_object(
        self, type_of_output, coreset_numbers, centers, depth, i, special_char=""
    ):
        """Loading the saved data

        Args:
            type_of_output: algorithm that created the outcome
            coreset_numbers: number of coresets
            centers: number of blobs
            depth: depth of a circuit
            df: data frame of the data
            i: hierachy iteration
            special_char (str, optional): Any special charcters that needed to be added to the file. Defaults to "".

        Returns:
            Raw data object
        """
        file_name = (
            self.data_folder
            + "/data/"
            + type_of_output
            + "/raw/"
            + type_of_output
            + "_df_coreset_"
            + str(coreset_numbers)
            + "_centers_"
            + str(centers)
            + "_p_"
            + str(depth)
            + "_i_"
            + str(i)
            + special_char
            + ".pickle"
        )
        open_file = open(file_name, "rb")
        loaded_list = pickle.load(open_file)
        open_file.close()

        return loaded_list

    def load_dendo_data(self, type_of_plot: str):
        """
        Load dendogram data
        Args:
            type_of_plot: algorithm that generated the plot
        Returns:
            dendrogram data
        """
        file_name = (
            self.data_folder
            + "/data/"
            + type_of_plot
            + "/"
            + type_of_plot
            + "_dendo_data_for_plot.pickle"
        )
        open_file = open(file_name, "rb")
        dendo_data = pickle.load(open_file)
        open_file.close()

        return dendo_data

    def load_dendo_coloring_data(self, type_of_plot: str):
        """Load data that saves color data
        Args:
            type_of_plot: algorithm that generated the data
        Returns:
            dendogram coloring data
        """
        file_name = (
            self.data_folder
            + "/data/"
            + type_of_plot
            + "/"
            + type_of_plot
            + "_dendo_color_dict.pickle"
        )
        open_file = open(file_name, "rb")
        coloring_dict = pickle.load(open_file)
        open_file.close()

        return coloring_dict

    def load_orqviz_params_dict(self, vqa_type: str):
        """
        load params related to orqviz
        Args:
            vqa_type: vqa name that created the data
        Returns:
            orqviz params
        """
        file_name = (
            self.data_folder
            + "data/"
            + vqa_type
            + "/"
            + vqa_type
            + "_cost_params_dict.pickle"
        )
        open_file = open(file_name, "rb")
        cost_params_dict = pickle.load(open_file)
        open_file.close()

        return cost_params_dict
