import pickle
import sys

import pandas as pd

from bigdatavqa.divisiveclustering import Dendrogram

method = sys.argv[1]

dendrogram_df = pd.read_pickle(
    f"data/results/divisive_clustering/dendrogram_{method}.pkl"
)


with open(f"data/results/divisive_clustering/hc_{method}_cut.pkl", "rb") as f:
    hc = pickle.load(f)

coreset_pd = pd.read_csv("data/25_coreset.csv")

with open("data/dataset.pickle", "rb") as f:
    raw_data = pickle.load(f)


if __name__ == "__main__":
    dendo = Dendrogram(hc, coreset_pd, raw_data)
    dendo.get_centroid_dist_df()
    dendo.get_cluster_reference_dict()

    dendo.plot_dendrogram(
        dendrogram_df=dendrogram_df,
        plot_name=f"dendrogram_{method}.png",
        # vertical_line=6,
    )
