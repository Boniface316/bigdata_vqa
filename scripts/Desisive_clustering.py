import argparse
import warnings

import cudaq
import numpy as np
import pandas as pd
from bigdatavqa.coreset import Coreset
from bigdatavqa.divisiveclustering import (
    DivisiveClusteringVQA,
)
from bigdatavqa.optimizer import get_optimizer_for_QAOA
from bigdatavqa.vqe_utils import get_K2_Hamiltonian, get_QAOA_circuit

warnings.filterwarnings("ignore")


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-t",
    "--target",
    type=str,
    default="qpp-cpu",
    choices=["qpp-cpu", "nvidia", "nvidia-mgpu"],
    help="Quantum simulator backend. Default is qpp-cpu. See https://nvidia.github.io/cuda-quantum/0.6.0/using/simulators.html for more options.",
)

argparser.add_argument(
    "-d",
    "--depth",
    type=int,
    default=1,
    help="Depth of the QAOA circuit. Default is 1.",
)
argparser.add_argument(
    "-i",
    "--max_iterations",
    type=int,
    default=75,
    help="Max iterations for the optimizer.",
)
argparser.add_argument(
    "-s", "--max_shots", type=int, default=100000, help="Max shots for the simulation."
)
argparser.add_argument("-m", "--M", type=int, default=10, help="Size of the coreset.")

argparser.add_argument(
    "-n", "--N", type=int, default=10000, help="Number of rows for the raw dataset"
)

argparser.add_argument(
    "--number_of_sampling_for_centroids",
    type=int,
    default=10,
    help="Number of sampling for the centroids.",
)

argparser.add_argument(
    "--number_of_coresets_to_evaluate",
    type=int,
    default=10,
    help="Number of coresets to evaluate.",
)

argparser.add_argument(
    "--coreset_method",
    type=str,
    default="BFL2",
    choices=["BFL2", "BLK2"],
    help="Method for the coreset creation.",
)

argparser.add_argument(
    "--normalize_vectors",
    type=bool,
    default=True,
    help="Normalize the vectors.",
)

argparser.add_argument(
    "--max_cutoff_threshold",
    type=float,
    default=0.5,
    help="Max cutoff threshold for the divisive clustering.",
)

argparser.add_argument(
    "--coreset_to_graph_metric",
    type=str,
    default="dist",
    choices=["dist", "dot"],
    help="Metric to calculate the weight of the graph.",
)

args = argparser.parse_args()

number_of_rows = args.N
target = args.target
coreset_size = args.M
circuit_depth = args.depth
max_iterations = args.max_iterations
max_shots = args.max_shots
max_cutoff_threshold = args.max_cutoff_threshold
coreset_to_graph_metric = args.coreset_to_graph_metric

number_of_sampling_for_centroids = args.number_of_sampling_for_centroids
number_of_coresets_to_evaluate = args.number_of_coresets_to_evaluate
coreset_method = args.coreset_method
normalize_vectors = args.normalize_vectors


def divisive_using_VQA(
    circuit_depth,
    max_iterations,
    max_shots,
    max_cutoff_threshold,
    coreset_to_graph_metric,
    get_K2_Hamiltonian,
    optimizer,
    normalize_vectors,
    optimizer_function,
    create_circuit,
):
    divisive_clustering = DivisiveClusteringVQA(
        circuit_depth=circuit_depth,
        max_iterations=max_iterations,
        max_shots=max_shots,
        threshold_for_max_cut=max_cutoff_threshold,
        create_Hamiltonian=get_K2_Hamiltonian,
        optimizer=optimizer,
        optimizer_function=optimizer_function,
        create_circuit=create_circuit,
        normalize_vectors=normalize_vectors,
        sort_by_descending=True,
        coreset_to_graph_metric=coreset_to_graph_metric,
    )

    return divisive_clustering.get_divisive_sequence(
        coreset_df, vector_columns=["X", "Y"]
    )


def main(methods, *args, **kwargs):
    cost_dict = {}
    for method in methods:
        if method == "QAOA":
           hierarchial_clustering_sequence =  divisive_using_QAOA(*args, **kwargs)
           cost = sum(
                divisive_clustering.get_divisive_cluster_cost(
                    hierarchial_clustering_sequence, coreset_df, vector_columns=["X", "Y"]
                )
            )
            cost_dict[method] = cost
        elif method == "VQE":



if __name__ == "__main__":
    cudaq.set_target(target)

    raw_data = Coreset.create_dataset(number_of_rows)
    coreset = Coreset(
        raw_data=raw_data,
        number_of_sampling_for_centroids=number_of_sampling_for_centroids,
        coreset_size=coreset_size,
        number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
        coreset_method=coreset_method,
    )

    coreset_vectors, coreset_weights = coreset.get_best_coresets()

    coreset_df = pd.DataFrame(
        {
            "X": coreset_vectors[:, 0],
            "Y": coreset_vectors[:, 1],
            "weights": coreset_weights,
        }
    )
    coreset_df["Name"] = [chr(i + 65) for i in coreset_df.index]

    optimizer = cudaq.optimizers.COBYLA()

    divisive_clustering = DivisiveClusteringVQA(
        circuit_depth=circuit_depth,
        max_iterations=max_iterations,
        max_shots=max_shots,
        threshold_for_max_cut=max_cutoff_threshold,
        create_Hamiltonian=get_K2_Hamiltonian,
        optimizer=optimizer,
        optimizer_function=get_optimizer_for_QAOA,
        create_circuit=get_QAOA_circuit,
        normalize_vectors=normalize_vectors,
        sort_by_descending=True,
        coreset_to_graph_metric=coreset_to_graph_metric,
    )

    hierarchial_clustering_sequence = divisive_clustering.get_divisive_sequence(
        coreset_df, vector_columns=["X", "Y"]
    )

    cost = sum(
        divisive_clustering.get_divisive_cluster_cost(
            hierarchial_clustering_sequence, coreset_df, vector_columns=["X", "Y"]
        )
    )

    print(f"Cost: {cost}")

    circuit = get_QAOA_circuit(5, circuit_depth)

    print(cudaq.draw(circuit, np.random.rand(2 * circuit_depth), 5, circuit_depth))
