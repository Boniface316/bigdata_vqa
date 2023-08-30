import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.optimize import minimize, LinearConstraint
from math import pi
import operator
import pandas as pd
import qiskit
from qiskit import Aer, QuantumCircuit, transpile, assemble
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram

qiskit.utils.algorithm_globals.random_seed = 0


class MaxCutQaoa:
    '''
    Class to implement the QAOA for the weighted MaxCut problem.
    Each vertex has an associated feature vector and the dot
    product between feature vectors of connecting vertices are
    evaluated. Input is the problem graph for which the approximate
    MaxCut solution will be found.
    '''
    def __init__(self, graph):
        self.graph = graph
        self.gamma = [0]
        self.beta = [0]
        self.p = int((len(self.gamma) + len(self.beta)) / 2)
        self.theta = np.concatenate((self.gamma, self.beta), axis=None)

    def problem_gate(self, circuit, gamma, node1, node2):
        '''
        Implements the two qubit gate:
        exp(i* gamma/2 * (Id - (Z_i x Z_j)) * w_i * w_j * (x_i, x_j)).
        Uses Control-Phase gate (CP-gate) followed by a phase gate on each
        gate with angle = -2*gamma*edge_weight used as the parameter for CP,
        and angle for the P gates.
        Unitary implemented:
           [[1,   0,            0            0],
            [0, exp(i*angle),   0,           0],
            [0,   0,          exp(i*angle),  0],
            [0,   0,            0,           1]]
        Params:
        circuit: type QuantumCircuit
        gamma: type float
        qbit1, qbit2: type qubits
        '''
        #Multiply edge weight with gamma for the rotation angle
        angle = gamma * self.graph[node1][node2]['weight']
        #Add to circuit
        circuit.cp(-2 * angle, node1, node2)
        circuit.p(angle, node1)
        circuit.p(angle, node2)



    def set_circuit(self):
        '''
        Constructs the QAOA circuit and stores it in the circuit attribute
        '''
        edge_list = list(self.graph.edges)
        no_vertices = len(list(self.graph.nodes))
        QAOA = QuantumCircuit(no_vertices, no_vertices)
        #Add layer of Hadamards
        QAOA.h(range(no_vertices))
        QAOA.barrier()

        #Add the alternating unitaries to the circuit
        for i in range(self.p):
            #Add the gates from the problem Hamiltonian
            for edge in edge_list:
                self.problem_gate(QAOA, self.gamma[i], edge[0], edge[1])
            QAOA.barrier()
            #Add the gates from the mixer Hamiltonian
            QAOA.rx(2 * self.beta[i], range(no_vertices))
            QAOA.barrier()

        #Measure in comp. basis and write out to classical bits
        QAOA.measure(range(no_vertices), range(no_vertices))

        self.circuit = QAOA

    def get_circuit(self):
        return self.circuit

    def prepare_circuit(self):
        self.set_circuit()

    def set_initial_params(self, params):
        '''
        Set the starting values for the parameters gamma and beta
        Inputs:
        params: type (1, 2) array containing the gamma then beta parameters.
        '''
        new_p = len(params) / 2
        assert new_p - int(new_p) == 0.0, 'Number of parameters must be a multiple of 2'

        int_p = int(new_p)
        self.gamma = params[:int_p]
        self.beta = params[int_p:]
        self.p = int_p
        self.theta = params

    def show_circuit(self):
        self.circuit.draw(output='mpl', style='iqx', fold=50)
        plt.show()

    def show_graph(self):
        # Generate plot of the Graph
        colors       = ['r' for node in self.graph.nodes()]
        default_axes = plt.axes(frameon=True)
        pos          = nx.spring_layout(self.graph)

        nx.draw_networkx(self.graph, node_color=colors, node_size=600,
                         alpha=1, ax=default_axes, pos=pos)
        plt.show()

    def run_circuit(self, shots):
        '''
        Assembles and runs the circuit and sets the results attribute.
        '''
        qobj = assemble(self.circuit)
        sim = Aer.get_backend('aer_simulator')
        sim.shots = shots
        result = sim.run(qobj).result()

        self.results = result
        self.counts = result.get_counts()

    def cost_function(self, x):
        '''
        Returns the value of the weighted MaxCut cost function
        evaluated at state x for the graph
        Params:
        G: type graph
        x: type bit-string/state
        '''
        edge_list = list(self.graph.edges)

        C = 0
        for edge in edge_list:
            #Extract the vertices that the edge connects and store them
            #in v_i and v_j
            v_i = edge[0]
            v_j = edge[1]
            #Convert the character '0' or '1' (as is used in the state
            #represenation of x) to an integer for use in the cost function.
            x_i = int(x[v_i])
            x_j = int(x[v_j])
            #Extract edge weight
            edge_weight = self.graph[v_i][v_j]['weight']
            #Compute cost
            C += (edge_weight * ((x_i * (1 - x_j)) + (x_j * (1 - x_i))))

        return C


    def compute_expectation(self):
            '''
            Returns the estimated expectation value of a trial state
            |thi(theta)>
            Params:
            G: type graph
            results: type dictionary
            The results parameter is the dictionary object containing
            the counts of each measured state (self.counts).
            '''
            no_results = 0
            exp_val = 0
            #Create list of computational basis states
            comp_states = list(self.counts.keys())
            for state in comp_states:
                #Multiply number of times the state is measured with
                #the cost function evaluated at that state.
                state_count = self.counts[state] * self.cost_function(state)
                exp_val = exp_val + state_count
                no_results = no_results + self.counts[state]

            self.exp_val = exp_val * (1 / no_results)

    def execute(self, shots=1024):
        '''
        Run this function to execute the algorithm and generate an
        approximate expectation value for the trial state |thi(theta)>
        '''
        self.prepare_circuit()
        self.run_circuit(shots)
        self.compute_expectation()

    def get_expectation(self):
        return self.exp_val

    @staticmethod
    def Qaoa_Algorithm(x, self, shots):
        #Helper method for optimising the parameters: returns negative
        #expectation value.
        self.set_initial_params(x)
        self.prepare_circuit()
        self.run_circuit(shots)
        self.compute_expectation()

        return -self.exp_val

    def optimise(self, shots=1024, initial_params=[0,0]):
        #Minimises the negative expectation value and returns optimal
        #parameter array. This is equivalent to maximising the positive
        #expectation value.
        p = len(initial_params) / 2
        assert p - int(p) == 0.0, 'Number of parameters must be a multiple of 2'
        cons1 = tuple({'type': 'ineq', 'fun': lambda x: x[i]} \
                                      for i in range(int(p)))
        cons2 = tuple({'type': 'ineq', 'fun': lambda x: (2 * pi) - x[i]} \
                                      for i in range(int(p)))
        cons = cons1 + cons2
        opts = minimize(self.Qaoa_Algorithm, initial_params, args=(self, shots),
                        method='COBYLA', constraints=cons)

        return opts

    def show_histogram(self):
        plot_histogram(self.results.get_counts())
        plt.show()

    def get_counts(self):
        return self.counts

    def get_optimal_state(self):
        '''
        Brute force searches the computational basis states for the
        state that maximises the cost function.
        '''
        cost_list = []
        opt_state_list = []
        #Create list of computational basis states
        comp_states = list(self.counts.keys())
        for state in comp_states:
            #Create list of the costs, ignoring states with no partition
            if '0' not in state:
                pass
            if '1' not in state:
                pass
            else:
                cost_list.append(self.cost_function(state))
        val, idx = max((val, idx) for (idx, val) in enumerate(cost_list))
        opt_state = comp_states[idx]
        print(f'Optimal state is {opt_state}, with cost = {val}')
        return opt_state


class mean_clusters:
    '''
    Finds cluster centres mu_1, mu_2 that approximately maximise
    the 2-means objective function with weighted cluster centres.
    Does so by using the mapping to a weighted MaxCut problem on
    a complete graph and uses the QAOA to find the optimal
    Inputs:
    data: type = pandas dataframe
    '''
    def __init__(self, data, weights=None):
        self.data = data
        self.weights = weights
        self.graph = self.data_to_graph()
        self.qaoa = MaxCutQaoa(self.graph)

    def data_to_graph(self):
        '''
        Transforms the input data (type: pandas dataframe with
        feature vectors as rows or numpy array) into a nx graph object with
        associated feature vectors and weights. The 2-means clustering
        problem maps to a weighted MaxCut on a complete graph, therefore
        every pair of vertices will be connected by an edge. Only vertex
        weights that cross the cut contribute to the cost.
        '''
        #Generate graph
        if isinstance(self.data, pd.DataFrame):
            feature_mat = self.data.to_numpy()
        else:
            feature_mat = self.data
        (no_vertices, _) = feature_mat.shape
        G = nx.complete_graph(no_vertices)

        #Add feature vectors to the vertices
        for i in range(no_vertices):
            G.add_node(i, feature_vector=feature_mat[i])

        edge_list = list(G.edges)
        #Compute and add edge weights to the graph
        if self.weights is None:
            for edge in edge_list:
                v_i = edge[0]
                v_j = edge[1]
                fv_i = G.nodes[v_i]['feature_vector']
                fv_j = G.nodes[v_j]['feature_vector']
                edge_weight = np.dot(fv_i, fv_j)
                G[v_i][v_j]['weight'] = edge_weight
        else:
            for edge in edge_list:
                v_i = edge[0]
                v_j = edge[1]
                fv_i = G.nodes[v_i]['feature_vector']
                fv_j = G.nodes[v_j]['feature_vector']
                dp = np.dot(fv_i, fv_j)
                edge_weight = self.weights[v_i] * self.weights[v_j] * dp
                G[v_i][v_j]['weight'] = edge_weight

        return G

    def approx_partition(self, params=[0,0]):
        '''
        Returns partitioning of the data into two sets based on the
        results of the MaxCut QAOA optimisation procedure. Partitioning
        is only an approximation and could potentially be poor. This
        could be improved by repeating the optimisation routine and
        selecting the best results. Equivalently this method could be called
        multiple times and the most consistent partition could be selected,
        or evaluating the objective function on each of the partitions.
        Inputs:
        params: type = array containing the 2p QAOA parameters. Default is
                p=1 and [0,0].
        '''
        opt_results = self.qaoa.optimise(initial_params=params)
        self.qaoa.set_initial_params(opt_results.x)
        self.qaoa.execute()
        state_counts = self.qaoa.counts
        approx_state = max(state_counts.items(),
                           key=operator.itemgetter(1))[0]
        partition = (list(map(int, approx_state)))
        S1 = []
        S2 = []
        for i in range(len(partition)):
            if partition[i] == 0:
                S1.append(i)
            else:
                S2.append(i)

        sets = []
        sets.append(S1)
        sets.append(S2)
        #Set the approx_state attribute to be the best state found
        #after the optimisation of the parameters.
        self.approx_state = approx_state
        return sets

    def approx_centres(self, params=[0,0], weight_type=None):
        '''
        Returns approximate cluster centres as a 2d array.
        NOTE: need to reimplement this to deal with weights
              to do so need to divide by W-1 or W+1 (total
              weights) when finding the mean clustering centre.
        '''
        partition = self.approx_partition(params=params)
        #Calculate length of feature vectors
        if isinstance(self.data, pd.DataFrame):
            cluster_size = len(self.data.to_numpy()[0])
        else:
            cluster_size = len(self.data[0])
        c1 = np.zeros(cluster_size)
        c2 = np.zeros(cluster_size)
        clusters = np.array([c1, c2])
        #For unweighted clustering problem - default
        if weight_type is None:
            for i in range(len(partition)):
                for vertex in partition[i]:
                    clusters[i] += self.graph.nodes[vertex]['feature_vector'] \
                                            * (1/len(partition[i]))
        #Weighted clusters assuming that W1=W2=W/2 i.e equal cluster weights
        elif weight_type == 'equal':
            W_half = np.sum(self.weights) * 0.5
            for i in range(len(partition)):
                for vertex in partition[i]:
                    weight = self.weights[vertex] * self.graph.nodes[vertex]['feature_vector']
                    clusters[i] += weight * (1/W_half)
        #Weighted clusters with no assumptions about equal clusters
        elif weight_type == 'not_equal':
            W = [0, 0]
            for i in range(len(partition)):
                for vertex in partition[i]:
                    W[i] += self.weights[vertex]
            for i in range(len(partition)):
                for vertex in partition[i]:
                    weight = self.weights[vertex] * self.graph.nodes[vertex]['feature_vector']
                    clusters[i] += weight * (1/W[i])

        self.partition = partition
        self.clusters = clusters
        return clusters

    def cluster_cost(self):
        '''
        Computes the cost of the approximate clusters found.
        '''
        clusters = self.clusters
        partition = self.partition
        cost = 0
        if self.weights is None:
            for vertex in range(len(self.graph.nodes)):
                if self.graph.nodes[vertex] in partition[0]:
                    cost += np.linalg.norm(self.graph.nodes[vertex]['feature_vector'] \
                                           - clusters[0]) ** 2
                else:
                    cost += np.linalg.norm(self.graph.nodes[vertex]['feature_vector'] \
                                           - clusters[1]) ** 2
        else:
            for vertex in range(len(self.graph.nodes)):
                if self.graph.nodes[vertex] in partition[0]:
                    dist =  (np.linalg.norm(self.graph.nodes[vertex]['feature_vector'] \
                                           - clusters[0]) ** 2)
                    cost += self.weights[vertex] * dist
                else:
                    dist = (np.linalg.norm(self.graph.nodes[vertex]['feature_vector'] \
                                           - clusters[1]) ** 2)
                    cost += self.weights[vertex] * dist

        return cost
