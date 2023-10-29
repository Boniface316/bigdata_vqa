from bigdatavqa.divisiveclustering import Dendrogram
import pickle
import sys
import pandas as pd

method = sys.argv[1]

dendrogram_df = pd.read_pickle(f"data/dendrogram_{method}.pkl")


with open(f"data/hc_{method}_cut.pkl", "rb") as f:
    hc = pickle.load(f)

coreset_pd = pd.read_csv("data/coreset.csv")

with open("data/dataset.pickle", "rb") as f:
    raw_data = pickle.load(f)


if __name__ == "__main__":
    dendo = Dendrogram(hc, coreset_pd, raw_data)
    dendo.get_centroid_dist_df()
    dendo.get_cluster_reference_dict()

    dendo.plot_dendrogram(
        dendrogram_df=dendrogram_df,
        plot_name=f"dendrogram_{method}.png",
        vertical_line=6,
    )
