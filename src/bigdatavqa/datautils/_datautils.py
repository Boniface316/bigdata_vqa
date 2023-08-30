import os
import pickle
from typing import Dict, List

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
        with open(f"{self.data_folder}/{file_name}", 'rb') as handle:
            X = pickle.load(handle)
            print(f"Data loaded from {self.data_folder}/{file_name}")
        return X







