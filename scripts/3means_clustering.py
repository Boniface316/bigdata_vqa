import argparse
import os as os

import pandas as pd
from loguru import logger

from bigdatavqa.coreset import Coreset, normalize_np
from bigdatavqa.datautils import DataUtils
from bigdatavqa.k3meansclustering import get_3means_cluster_centers_and_cost

parser = argparse.ArgumentParser(description="GMM experiment parameters")

parser.add_argument("--qubits", type=int, required=True, help="Number of qubits")
parser.add_argument("--circuit_depth", type=int, required=True, help="Circuit depth")
parser.add_argument(
    "--number_of_shots", type=int, required=True, help="Number of shots"
)
parser.add_argument(
    "--iterations", type=int, required=True, help="Number of iterations"
)
parser.add_argument(
    "--number_of_experiment_runs",
    type=int,
    required=False,
    default=5,
    help="Number of times to run the experiment",
)
parser.add_argument(
    "--data_location", type=str, required=False, default="data", help="Data location"
)
parser.add_argument(
    "--number_of_coresets_to_evaluate",
    type=int,
    default=10,
    required=False,
    help="Number of coresets to analyze for the best",
)
parser.add_argument(
    "--centroid_numbers",
    type=int,
    default=10,
    help="Number of times to run to find coreset centers",
)
args = parser.parse_args()


logger.add(
    ".logs/3means_clustering.log",
    rotation="10 MB",
    compression="zip",
    level="INFO",
    retention="10 days",
)


number_of_qubits = args.qubits
circuit_depth = args.circuit_depth
max_shots = args.number_of_shots
max_iterations = args.iterations
number_of_experiment_runs = args.number_of_experiment_runs
data_location = args.data_location
number_of_corsets_to_evaluate = args.number_of_coresets_to_evaluate
number_of_centroid_evaluation = args.centroid_numbers

logger.info(f"Number of qubits: {number_of_qubits}")
logger.info(f"Circuit depth: {circuit_depth}")
logger.info(f"Max iterations: {max_iterations}")
logger.info(f"Max shots: {max_shots}")
logger.info(f"Number of experiments to run: {number_of_experiment_runs}")
logger.info(f"Data location: {data_location}")
logger.info(f"Number of coresets to evaluate: {number_of_corsets_to_evaluate}")
logger.info(f"Number of centroid evaluations: {number_of_centroid_evaluation}")


def get_raw_data(data_location):
    return raw_data


# Means3Vqe = Vqe_3_Means(data, sample_size=5)
# results = Means3Vqe.fit_coreset([1, 2])
# results


if __name__ == "__main__":
    data_utils = DataUtils(data_location)

    try:
        raw_data = data_utils.load_dataset()
    except FileNotFoundError:
        raw_data = data_utils.create_dataset(n_samples=1000)

    coreset = Coreset()
    coreset_vectors, coreset_weights = coreset.get_best_coresets(
        data_vectors=raw_data,
        number_of_runs=size_vec_list,
        coreset_numbers=number_of_qubits,
        use_kmeans_cost=False,
    )

    coreset_vectors, coreset_weights = normalize_np(
        coreset_vectors, centralize=True
    ), normalize_np(coreset_vectors, centralize=True)

    cluster_centers, cluster_cost = get_3means_cluster_centers_and_cost(
        coreset_vectors,
        coreset_weights,
        circuit_depth,
        raw_data,
        num_runs=max_iterations,
    )
