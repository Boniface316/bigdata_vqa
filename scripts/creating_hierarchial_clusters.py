from bigdatavqa.divisiveclustering import Dendrogram
import pickle
import sys
import pandas as pd

current_position = int(sys.argv[1])

method = sys.argv[2]


with open(f"data/hc_{method}_cut.pkl", "rb") as f:
    hc = pickle.load(f)

coreset_pd = pd.read_csv("data/coreset.csv")

with open("data/dataset.pickle", "rb") as f:
    raw_data = pickle.load(f)

if __name__ == "__main__":
    dendo = Dendrogram(hc, coreset_pd, raw_data)

    dendo.plot_dendrogram_manually(current_position)
