from bigdatavqa.datautils import DataUtils
from bigdatavqa.divisiveclustering.divisiveclustering import get_coreset_vec_and_weights
from bigdatavqa.coreset import normalize_np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_coreset_pd():
    data_location = "data"
    data_utils = DataUtils(data_location)
    try:
        raw_data = data_utils.load_dataset()
    except FileNotFoundError:
        raw_data = data_utils.create_dataset(n_samples=1000)

    coreset_vec, coreset_weights = get_coreset_vec_and_weights(raw_data, 25, 10, 20)

    coreset_pd = pd.DataFrame(coreset_vec)
    coreset_pd["weights"] = coreset_weights
    coreset_pd["name"] = [chr(i + 65) for i in range(25)]

    coreset_pd = coreset_pd.rename(columns={0: "X", 1: "Y"})

    coreset_vec_normalized = normalize_np(coreset_vec, centralize=True)
    coreset_weights_normalized = normalize_np(coreset_weights, centralize=True)

    coreset_pd["X_norm"] = coreset_vec_normalized[:, 0]
    coreset_pd["Y_norm"] = coreset_vec_normalized[:, 1]
    coreset_pd["weights_norm"] = coreset_weights_normalized

    coreset_pd.to_csv("coreset_normalized.csv")


try:
    coreset_pd = pd.read_csv("coreset_normalized.csv")
except FileNotFoundError:
    create_coreset_pd()
    coreset_pd = pd.read_csv("coreset_normalized.csv")


plt.scatter(coreset_pd["X"], coreset_pd["Y"], s=coreset_pd["weights"])
plt.xlabel("X")
plt.ylabel("Y")
# BEGIN: zy8f9d3g4h5j
for i, row in coreset_pd.iterrows():
    x = row["X"]
    y = row["Y"]
    name = row["name"]
    plt.annotate(name, (x, y))

plt.savefig("coreset.png")

plt.clf()


plt.scatter(
    coreset_pd["X_norm"], coreset_pd["Y_norm"], s=coreset_pd["weights_norm"] * 10
)
plt.xlabel("X")
plt.ylabel("Y")
for i, row in coreset_pd.iterrows():
    x = row["X_norm"]
    y = row["Y_norm"]
    name = row["name"]
    plt.annotate(name, (x, y))

plt.savefig("coreset_norm.png")
plt.clf()
