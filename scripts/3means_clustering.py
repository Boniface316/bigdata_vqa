import argparse
import os as os

import pandas as pd
from loguru import logger

from bigdatavqa.coreset import Coreset
from bigdatavqa.datautils import DataUtils

# from meansClustering.vqe3means import Vqe_3_Means


parser = argparse.ArgumentParser(description="Divisive clustering circuit parameters")

parser.add_argument("--qubits", type=int, required=True, help="Number of qubits")
parser.add_argument("--depth", type=int, required=True, help="Number of layers")
parser.add_argument("--shots", type=int, required=True, help="Number of shots")
parser.add_argument(
    "--iterations", type=int, required=True, help="Number of iterations"
)
parser.add_argument("--data_location", type=str, required=False, help="Data location")
args = parser.parse_args()


logger.add(
    ".logs/3means_clustering.log",
    rotation="10 MB",
    compression="zip",
    level="INFO",
    retention="10 days",
)


number_of_qubits = args.qubits
circuit_depth = args.depth
max_shots = args.shots
max_iterations = args.iterations
if args.data_location is None:
    data_location = "data"
else:
    data_location = args.data_location

max_iterations = 100
number_of_runs = 100
size_vec_list = 10


def get_raw_data(data_location):
    data_utils = DataUtils(data_location)

    try:
        raw_data = data_utils.load_dataset()
    except FileNotFoundError:
        raw_data = data_utils.create_dataset(n_samples=1000)

    return raw_data


# Means3Vqe = Vqe_3_Means(data, sample_size=5)
# results = Means3Vqe.fit_coreset([1, 2])
# results


if __name__ == "__main__":
    raw_data = get_raw_data(data_location)
    coreset = Coreset()
    coreset_vectors, coreset_weights = coreset.get_best_coresets(
        data_vectors=raw_data,
        number_of_runs=number_of_runs,
        coreset_numbers=number_of_qubits,
        use_kmeans_cost=False,
    )
