import argparse
import os as os

import numpy as np
from loguru import logger

from bigdatavqa.coreset import Coreset, normalize_np
from bigdatavqa.datautils import DataUtils
from bigdatavqa.k3meansclustering import get_3means_cluster_centers_and_cost

best_cost = np.inf
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


coreset_size = args.qubits * 2
circuit_depth = args.circuit_depth
max_shots = args.number_of_shots
max_iterations = args.iterations
number_of_experiment_runs = args.number_of_experiment_runs
data_location = args.data_location
number_of_corsets_to_evaluate = args.number_of_coresets_to_evaluate
number_of_centroid_evaluation = args.centroid_numbers

logger.info(f"Coreset size: {coreset_size}")
logger.info(f"Circuit depth: {circuit_depth}")
logger.info(f"Max iterations: {max_iterations}")
logger.info(f"Max shots: {max_shots}")
logger.info(f"Number of experiments to run: {number_of_experiment_runs}")
logger.info(f"Data location: {data_location}")
logger.info(f"Number of coresets to evaluate: {number_of_corsets_to_evaluate}")
logger.info(f"Number of centroid evaluations: {number_of_centroid_evaluation}")


if __name__ == "__main__":
    data_utils = DataUtils(data_location)

    try:
        raw_data = data_utils.load_dataset()
    except FileNotFoundError:
        raw_data = data_utils.create_dataset(n_samples=1000)

    for i in range(number_of_experiment_runs):
        cluster_centers, cost_for_clusters = get_3means_cluster_centers_and_cost(
            raw_data,
            circuit_depth,
            coreset_size,
            number_of_corsets_to_evaluate,
            number_of_centroid_evaluation,
            max_shots,
            max_iterations,
        )
        logger.info(
            f"Clusters:{cluster_centers} \n  Cost for clusters: {cost_for_clusters}"
        )

        if cost_for_clusters < best_cost:
            best_cost = cost_for_clusters
            best_cluster_centers = cluster_centers
            logger.info("New cost is better than the previous best cost")

    logger.success(
        f"Best cluster centers: {best_cluster_centers} \n Best cost: {best_cost}"
    )
