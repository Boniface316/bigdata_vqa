import pandas as pd

from bigdatavqa.coreset import Coreset


number_of_centroids_evaluation = 10
number_of_coresets_to_evaluate = 20
number_of_qubits = 10

raw_data = pd.read_csv("data/dataset.pickle")


coreset = Coreset()

initial_coreset_vectors, initial_coreset_weights = coreset.get_best_coresets(
    data_vectors=raw_data,
    number_of_runs=number_of_centroids_evaluation,
    coreset_size=number_of_qubits,
    number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
)


kernel = get_kernel()

optimizer = get_optimizer()


hierarchical_clustering = DivisiveClustering(