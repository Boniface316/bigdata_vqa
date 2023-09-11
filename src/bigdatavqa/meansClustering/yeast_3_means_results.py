import pandas as pd
import numpy as np
import coreset as cs
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import vqe
from sklearn.cluster import KMeans

'''
Load data and build the coreset and problem graph
'''
data =  pd.read_csv('datasets/yeast.data', sep=r'\s+', header=None)

yeast_names = data[0]
data.drop(0, axis=1, inplace=True)

labels = data[9]
data.drop(9, axis=1, inplace=True)

#Convert to numpy matrix
data = data.to_numpy()


'''
Parameter values
'''
depth = [1, 2, 5, 10]
coreset_size = [8,9]



vqe_simulations_coreset = np.zeros((4, 4))
vqe_bounds_coreset = np.zeros(4)

vqe_simulations_random_sample = np.zeros((2, 4))
vqe_bounds_random_sample = np.zeros(2)


# for i, size in enumerate(coreset_size):
#     coreset, weights = cs.Algorithm2(data, 3, B, size)
#     coreset = np.array(coreset)
#     weights = np.array(weights)
#     G = vqe.data_to_graph(coreset, weights=weights)
#     #Compute vqe bound on the coreset
#     vqe_bound_centres = vqe.best_clusters(G, data, weights=weights)
#     vqe_bound_cost = vqe.cluster_cost_whole_set(data, vqe_bound_centres)
#     vqe_bounds_coreset[i] = vqe_bound_cost
#     print(vqe_bounds_coreset)
#     for j, circuit_depth in enumerate(depth):
#         #Approximate vqe simulation
#         vqe_simulation_cost = vqe.approximate_n_trials(G, data, coreset, weights, circuit_depth, 5)
#         vqe_simulations_coreset[i][j] = vqe_simulation_cost
#         print(vqe_simulations_coreset)

class Vqe_3_Means:
    def __init__(self, data, sample_size=10, coreset=None, weights=None, random_sample=None):
        self.data = data
        self.sample_size = sample_size
        self.coreset = coreset
        self.weights = weights
        self.random_sample = random_sample

    def set_coreset_and_weights(self):
        '''
        returns the coreset and corresponding weights given the raw data
        '''
        B = cs.get_bestB(self.data, num_runs=100, k=3)
        coreset, weights = cs.Algorithm2(self.data, 3, B, self.sample_size)
        self.coreset = np.array(coreset)
        self.weights = np.array(weights)

    def set_graph_representation(self, weights=False):
        '''
        sets the networkx graph representation of the provided 
        data subset - coreset or random sample. 
        '''
        if weights:
            self.coreset_graph = vqe.data_to_graph(self.coreset, weights=self.weights)
            return 
        
        self.random_sample_graph = vqe.data_to_graph(self.random_sample)
        
    def set_best_cluster_centres_by_brute_force(self, weights=False):
        '''
        finds the optimal cluster centres on the coreset by brute force.
        takes the graph representation of the coreset, raw data and coreset
        weights as params 
        '''
        if weights:
            self.best_coreset_centres = vqe.best_clusters(self.coreset_graph, self.data, weights=self.weights)
            return
        
        self.best_random_sample_centres = vqe.best_clusters(self.random_sample_graph, self.data)
        
    def get_3_means_cost(self, centres):
        '''
        evaluates the 3-means cost function using the provided cluster centres.
        the data parameter is the dataset to evaluate the cost function on.
        '''
        return vqe.cluster_cost_whole_set(self.data, centres)

    def get_vqe_bound(self, weights=False):
        '''
        computes the bound on the whole dataset
        '''
        if weights:
            self.set_best_cluster_centres_by_brute_force(weights=True)
            return self.get_3_means_cost(self.best_coreset_centres)
        
        self.set_best_cluster_centres_by_brute_force()
        return self.get_3_means_cost(self.best_random_sample_centres)

    def get_vqe_simulation_results(self, depth, num_runs,  weights=False):
        if weights:
            return vqe.approximate_n_trials(self.coreset_graph, self.data, self.coreset, self.weights, depth, num_runs)
        
        return vqe.random_n_trials(self.random_sample_graph, self.data, self.random_sample, num_runs, depth)
        
    def get_classical_cost(data, num_cluster_centres=3):
        '''
        uses sci-kit learn library to compute the 3 mean cost
        '''
        kmeans = KMeans(n_clusters=num_cluster_centres, random_state=0).fit(data)
        return vqe.cluster_cost_whole_set(data, kmeans.cluster_centers_)
        
    def fit_coreset(self, circuit_depths, num_runs=5, coreset=None, weights=None):
        '''
        Approximates the best cluster centres on the coreset. 
        The return type is a dictionary with the simulation results
        '''
        if (self.coreset == None) and (self.weights == None):
            if (coreset == None) and (weights == None):
                self.set_coreset_and_weights()
            else:
                self.coreset = coreset
                self.weights = weights
        
        #Generate graph, then find best clusters and compute bound
        self.set_graph_representation(weights=True)

        print('Computing VQE bound...')
        vqe_bound = self.get_vqe_bound(weights=True)

        print(f'VQE bound: {vqe_bound}')

        total_depths = len(circuit_depths)
        simulation_costs = np.zeros(total_depths)
        simulation_centres = np.zeros((total_depths, 3))

        #VQE simulations for each circuit depth
        for i, depth in circuit_depths:
            print(f'[{i}/{total_depths}] Simulating with circuit depth {depth}')
            (simulation_costs[i], simulation_centres[i]) = self.get_vqe_simulation_results(depth, num_runs, weights=True)
            print(f'VQE simulation cost for circuit depth {depth}: {simulation_costs[i]}')

        return {
            "coreset": self.coreset,
            "weights": self.weights,
            "circuit depths": circuit_depths,
            "vqe costs": simulation_costs,
            "cluster centres": simulation_centres
        }









        
        
 

def get_vqe_bound_on_coreset(coreset_size):
    for i, size in enumerate(coreset_size):
        coreset, weights = cs.Algorithm2(data, 3, B, size)
        coreset = np.array(coreset)
        weights = np.array(weights)
        G = vqe.data_to_graph(coreset, weights=weights)
        #Compute vqe bound on the coreset
        vqe_bound_centres = vqe.best_clusters(G, data, weights=weights)
        vqe_bound_cost = vqe.cluster_cost_whole_set(data, vqe_bound_centres)
        vqe_bounds_coreset[i] = vqe_bound_cost
        print(vqe_bounds_coreset)
        for j, circuit_depth in enumerate(depth):
            #Approximate vqe simulation
            vqe_simulation_cost = vqe.approximate_n_trials(G, data, coreset, weights, circuit_depth, 5)
            vqe_simulations_coreset[i][j] = vqe_simulation_cost
            print(vqe_simulations_coreset)

#Find random sample costs
for i, size in enumerate(coreset_size):
    #Create random samples
    sample = data[np.random.choice(data.shape[0], size, replace=False), :]
    sample_graph = vqe.data_to_graph(sample, weights=None)
    bound_centres = vqe.best_clusters(sample_graph, data, weights=None)
    bound_cost = vqe.cluster_cost_whole_set(data, bound_centres)
    vqe_bounds_random_sample[i] = bound_cost
    print(vqe_bounds_random_sample)
    for j, circuit_depth in enumerate(depth):
        sim_cost = vqe.random_n_trials(sample_graph, data, sample, 5, circuit_depth)
        vqe_simulations_random_sample[i][j] = sim_cost
        print(vqe_simulations_random_sample)

vqe_sims_d1 =  np.append(vqe_simulations_random_sample[:,0], vqe_simulations_coreset[:,0])
vqe_sims_d2 =  np.append(vqe_simulations_random_sample[:,1], vqe_simulations_coreset[:,1])
vqe_sims_d3 =  np.append(vqe_simulations_random_sample[:,2], vqe_simulations_coreset[:,2])
vqe_sims_d4 =  np.append(vqe_simulations_random_sample[:,3], vqe_simulations_coreset[:,3])


vqe_bounds = np.append(vqe_bounds_random_sample, vqe_bounds_coreset)

a_file = open("results.txt", "w")
np.savetxt(a_file, vqe_sims_d1)
np.savetxt(a_file, vqe_sims_d2)
np.savetxt(a_file, vqe_sims_d3)
np.savetxt(a_file, vqe_sims_d4)
np.savetxt(a_file, vqe_bounds)
a_file.close()

kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
sci_kit_centres = kmeans.cluster_centers_
sci_kit_cost = vqe.cluster_cost_whole_set(data, sci_kit_centres)

index = [r'$m=6$', r'$m=7$', r'$m=8$', r'$m=9$', r'$m=6$', r'$m=7$', r'$m=8$', r'$m=9$']
df1 = pd.DataFrame({'VQE bound': vqe_bounds,
                   'VQE simulation': vqe_sims_d1}, index=index)
df2 = pd.DataFrame({'VQE bound': vqe_bounds,
                   'VQE simulation': vqe_sims_d2}, index=index)
df3 = pd.DataFrame({'VQE bound': vqe_bounds,
                   'VQE simulation': vqe_sims_d3}, index=index)
df4 = pd.DataFrame({'VQE bound': vqe_bounds,
                   'VQE simulation': vqe_sims_d4}, index=index)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10), sharey=True)
df1.plot.bar(rot=0, ax=ax[0][0])
df2.plot.bar(rot=0, ax=ax[0][1], legend=False)
df3.plot.bar(rot=0, ax=ax[1][0], legend=False)
df4.plot.bar(rot=0, ax=ax[1][1], legend=False)
ax[0][0].axhline(y=sci_kit_cost, linestyle='--', color='black')
ax[0][1].axhline(y=sci_kit_cost, linestyle='--', color='black')
ax[1][0].axhline(y=sci_kit_cost, linestyle='--', color='black')
ax[1][1].axhline(y=sci_kit_cost, linestyle='--', color='black')
ax[0][0].set_ylabel('Cost (lower is better)')
ax[1][0].set_ylabel('Cost (lower is better)')
ax[0][0].text(0,-9, 'Random samples')
ax[0][0].text(2.2, -9, 'Coresets')
'''
x_offset = -0.1
y_offset = 0.5
for p in ax[0][0].patches:
    b = p.get_bbox()
    val = "{:.1f}".format(b.y1 + b.y0)
    ax[0][0].annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
'''

ax[0][0].set_title('Depth = 1')
ax[0][1].set_title('Depth = 2')
ax[1][0].set_title('Depth = 5')
ax[1][1].set_title('Depth = 10')

plt.savefig('VQE_graphs_3-means.pdf')
plt.show()
