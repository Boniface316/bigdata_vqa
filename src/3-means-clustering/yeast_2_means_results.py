import pandas as pd
import numpy as np
import coreset as cs
from MaxCutQaoa import MaxCutQaoa, mean_clusters
from math import pi
from sklearn.cluster import KMeans
import qiskit
import matplotlib.pyplot as plt

#np.random.seed(42)
#qiskit.utils.algorithm_globals.random_seed = 0

def whole_data_cost(data, centres):
    '''
    Evaluate the two means cost function on the whole dataset.
    Inputs:
    data: type numpy array - the feature matrix to evaluate the cost against.
    centres: type array - the cluster centres to find the cost for.
    '''
    (no_rows, _) = data.shape
    centre1 = centres[0]
    centre2 = centres[1]
    cost = 0
    for i in range(no_rows):
        dist_to_1 = np.linalg.norm(data[i] - centre1) ** 2
        dist_to_2 = np.linalg.norm(data[i] - centre2) ** 2
        if dist_to_1 < dist_to_2:
            cost += dist_to_1
        else:
            cost += dist_to_2

    return cost

def sci_kit_cost(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    sci_kit_centres = kmeans.cluster_centers_

    cost = whole_data_cost(data, sci_kit_centres)

    return cost

def get_best_clustering_coreset(coreset, weights, num_runs, histogram=False):
    clustering = mean_clusters(coreset, weights=weights)
    best_cost = np.inf
    #Vary starting parameters
    gamma = np.random.uniform(0, 2 * pi, num_runs)
    beta = np.random.uniform(0, 2 * pi, num_runs)
    for i in range(num_runs):
        centres = clustering.approx_centres(params=[gamma[i], beta[i]], weight_type='equal')
        cost = clustering.cluster_cost()
        if cost < best_cost:
            best_cost = cost
            best_centres = centres
    if histogram:
        clustering.qaoa.show_histogram()
    else:
        pass
    return best_cost, best_centres

def get_best_clustering_random_sample(data, num_runs):
    clustering = mean_clusters(data)
    best_cost = np.inf
    #Vary starting parameters
    gamma = np.random.uniform(0, 2 * pi, num_runs)
    beta = np.random.uniform(0, 2 * pi, num_runs)
    for i in range(num_runs):
        centres = clustering.approx_centres(params=[gamma[i], beta[i]])
        cost = clustering.cluster_cost()
        if cost < best_cost:
            best_cost = cost
            best_centres = centres

    return centres


def qa_cost_coreset(data, size, k, num_runs=10):
    '''
    Calculates the cost on the whole dataset that the quantum
    algorithm yields.
    '''
    B = cs.get_bestB(data, num_runs=100, k=k)

    coreset, weights = cs.Algorithm2(data, k, B, size)
    coreset = np.array(coreset)
    weights = np.array(weights)

    cost, centres = get_best_clustering_coreset(coreset, weights, num_runs, histogram=False)
    cost_on_whole_set = whole_data_cost(data, centres)

    return cost_on_whole_set

def cost_random_sample(data, num_runs, k, size):
    sample = data[np.random.choice(data.shape[0], size, replace=False), :]
    centres = get_best_clustering_random_sample(sample, num_runs)

    cost_on_whole_set = whole_data_cost(data, centres)

    return cost_on_whole_set

def get_costs(data):
    k = 2
    data_sizes = [1, 5, 10]
    costs = []

    sci_kit_result = sci_kit_cost(data, 2)
    costs.append(sci_kit_result)

    for size in data_sizes:
        qaoa_sample_cost = cost_random_sample(data, 10, 2, size)
        costs.append(qaoa_sample_cost)
    for i in data_sizes:
        qaoa_coreset_cost = qa_cost_coreset(data, size, 2, num_runs=10)
        costs.append(qaoa_coreset_cost)

    return costs

'''
First load the data
'''

data =  pd.read_csv('datasets/yeast.data', sep=r'\s+', header=None)

yeast_names = data[0]
data.drop(0, axis=1, inplace=True)

labels = data[9]
data.drop(9, axis=1, inplace=True)

#Convert to numpy matrix
data = data.to_numpy()

'''
Generate results
'''

costs = get_costs(data)

#array = [10, 50, 40, 30, 40, 30, 20]
index = ['Classical', r'$m = 5$', r'$m = 10$', r'$m = 20$', r'$m = 5$', r'$m = 10$', r'$m = 20$']
df = pd.DataFrame({'lab': index, 'val': costs})
color = ['blue', 'green', 'green', 'green', 'red', 'red', 'red']

ax = df.plot.bar(x='lab', y='val', rot=0, color=color, legend=False)
x_axis = ax.xaxis
x_axis.label.set_visible(False)
ax.set_ylabel('Cost (lower is better)')
x_offset = -0.26
y_offset = 0.5
for p in ax.patches:
    b = p.get_bbox()
    val = "{:.2f}".format(b.y1 + b.y0)
    ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))


plt.show()
