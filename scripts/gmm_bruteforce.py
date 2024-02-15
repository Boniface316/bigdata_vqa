import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture

coreset = pd.read_csv("data/25_coreset.csv")

weights = coreset["weights"].to_numpy()
coreset = coreset[["X", "Y"]].to_numpy()

all_costs = []


def get_weighted_mean(coreset, weights):
    dim = len(coreset[0])
    size_coreset = len(coreset)
    mu = np.zeros(dim)
    for i in range(size_coreset):
        mu += coreset[i] * weights[i]
    return mu / sum(weights)


def get_weighted_scatter_matrix(coreset, weights):
    dim = len(coreset[0])
    size_coreset = len(coreset)
    T = np.zeros((dim, dim))
    mu = get_weighted_mean(coreset, weights)
    for i in range(size_coreset):
        T += weights[i] * np.outer((coreset[i] - mu), (coreset[i] - mu))

    return T


def cost_approximate_weighted(coreset, weights, S0, S1):
    w0 = sum(weights[S0])
    w1 = sum(weights[S1])
    W = sum(weights)

    w1_div_w0 = 3 - 4 * w0 / W
    w0_div_w1 = 3 - 4 * w1 / W

    T_inv = inv(get_weighted_scatter_matrix(coreset, weights))
    cost1 = 0
    for i in S0:
        cost1 += (
            w1_div_w0 * weights[i] ** 2 * np.dot(coreset[i], np.dot(T_inv, coreset[i]))
        )
    for i in S1:
        cost1 += (
            w0_div_w1 * weights[i] ** 2 * np.dot(coreset[i], np.dot(T_inv, coreset[i]))
        )
    cost2 = 0
    for j in range(len(coreset)):
        for i in range(j):
            if i in S0 and j in S0:
                cost2 += (
                    2
                    * w1_div_w0
                    * weights[i]
                    * weights[j]
                    * np.dot(coreset[i], np.dot(T_inv, coreset[j]))
                )
            if i in S1 and j in S1:
                cost2 += (
                    2
                    * w0_div_w1
                    * weights[i]
                    * weights[j]
                    * np.dot(coreset[i], np.dot(T_inv, coreset[j]))
                )
            else:
                cost2 += -2 * np.dot(coreset[i], np.dot(T_inv, coreset[j]))

    return -(cost1 + cost2)


def random_partition(coreset):
    n = len(coreset)
    S0 = np.random.choice(n, n // 2, replace=False)
    S1 = [i for i in range(n) if i not in S0]
    return S0, S1


def brute_force(coreset, weights):
    for n in range(2 ** len(coreset)):
        print(n)
        S0 = [
            idx
            for idx, value in enumerate(list(bin(n)[2:].zfill(len(coreset))))
            if value == "0"
        ]
        S1 = [
            idx
            for idx, value in enumerate(list(bin(n)[2:].zfill(len(coreset))))
            if value == "1"
        ]
        all_costs += [cost_approximate_weighted(coreset, weights, S0, S1)]

    return all_costs


if __name__ == "__main__":
    S0, S1 = random_partition(coreset)
    print(cost_approximate_weighted(coreset, weights, S0, S1))
    gmm = GaussianMixture(n_components=2)
    partition = gmm.fit_predict(coreset)
    S0 = [i for i in range(len(partition)) if partition[i] == 0]
    S1 = [i for i in range(len(partition)) if partition[i] == 1]
    print(cost_approximate_weighted(coreset, weights, S0, S1))
