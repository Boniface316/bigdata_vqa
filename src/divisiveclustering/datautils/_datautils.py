import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pickle
from scipy.stats import multivariate_normal


class DataUtils:
    def __init__(self, data_folder: str = None, random_seed: int = 1000):
        if data_folder is None:
            self.data_folder = os.getcwd()
        else:
            self.data_folder = data_folder

        self.random_seed = random_seed

    def create_dataset(
        self,
        n_samples: int,
        covariance_values: List[float] = [-0.8, -0.8],
        save_file: bool = True,
        n_features: int = 2,
        number_of_samples_from_distribution: int = 500,
        file_name : str = "data.pickle",
        mean_array: np.ndarray = np.array([[0,0], [7,1]])

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

        random_seed=self.random_seed
    
        X = np.zeros((n_samples,n_features))
    
        # Iterating over different covariance values
        for idx, val in enumerate(covariance_values):
            
            covariance_matrix = np.array([[1, val], [val, 1]])
            
             # Generating a Gaussian bivariate distribution
             # with given mean and covariance matrix
            distr = multivariate_normal(cov = covariance_matrix, mean = mean_array[idx], seed = random_seed)
            
            # Generating 500 samples out of the
            # distribution
            data = distr.rvs(size = number_of_samples_from_distribution)
            
            X[number_of_samples_from_distribution*idx:number_of_samples_from_distribution*(idx+1)][:] = data

        if save_file:
            # check if the data folder exists, if not create it
            if not os.path.exists(self.data_folder + "/data/"):
                os.makedirs(self.data_folder + "/data/")
            # save X as a pickle file in the data folder
            with open(f"{self.data_folder}/{file_name}", 'wb') as handle:
                pickle.dump(X, handle)
                print(f"Data saved in {self.data_folder}/{file_name}")
            
        return X

    def load_dataset(self, file_name:str = "dataset.pickle"):

        """
        Load a dataset

        Args:
            file_name (str, optional): file name of the dataset. Defaults to "dataset.pickle".

        Returns:
            loaded dataset
        """
        # split the file name to get the extension. If the extension is not pickle, raise an error
        file_extension = file_name.split(".")[-1]
        
        if file_extension == "pickle":
            with open(f"{self.data_folder}/{file_name}", 'rb') as handle:
                X = pickle.load(handle)
                print(f"Data loaded from {self.data_folder}/{file_name}")
        elif file_extension == "npy":
            X = np.load(f"{self.data_folder}/{file_name}")
        else:
            raise ValueError("File extension not supported. Please use pickle or npy")
        return X


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
    ) -> None:
        """
            Save coresets

        Args:
            best_coreset_vectors: Coreset vectors that we want to save
            best_coreset_weights: Coreset weights that we want to save
            coreset_numbers: Number of coresets
            centers: number of blobs
        """

        coreset_vector_file = f"{self.data_folder}/data/{coreset_numbers}_coreset_vectors.npy"
        
        coreset_weight_file = f"{self.data_folder}/data/{coreset_numbers}_coreset_weights.npy"

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

    def _save_VQE(self, hc : list):
        parent_directory = f"{self.data_folder}/data/VQE"
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
        file_name = f"{parent_directory}/VQE_data.pickle"
        with open(file_name, "wb") as handle:
            pickle.dump(hc, handle)
            print(f"Data saved in {file_name}")

    def _load_VQE(self,file_name: str = None):
        if file_name is None:
            file_name = f"{self.data_folder}/data/VQE/VQE_data.pickle"
        with open(file_name, "rb") as handle:
            hc = pickle.load(handle)
            print(f"Data loaded from {file_name}")
        return hc

    def save_object(
        self,
        type: str,
        hc: list,
        coreset_numbers: int = None,
        centers: int = None,
        depth: int = None,
        df: pd.DataFrame = None,
        str_prob_dict: Dict = None,
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
        if type == "VQE":
            self._save_VQE(hc)

        else:

            
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
        obj_type: Optional[str] = "hc",
        coreset_numbers: Optional[int] = None,
        centers: Optional[int] =None,
        depth: Optional[int] = None,
        
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

            if output_type == "VQE":
                return self._load_VQE()
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
