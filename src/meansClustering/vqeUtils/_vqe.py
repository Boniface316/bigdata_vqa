from qiskit import Aer, QuantumCircuit, transpile, assemble
from qiskit import QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import itertools
from math import pi
from scipy.optimize import minimize, LinearConstraint
import operator
#np.random.seed(42)

'''
num_qubits = 4
# the rotation gates are chosen randomly, so we set a seed for reproducibility
ansatz = EfficientSU2(num_qubits, reps=1, entanglement='full', insert_barriers=True)
ansatz.draw('mpl', style='iqx')
plt.show()

print(list(ansatz.parameters))
'''
def data_to_graph(data, weights=None):
    '''
    Transforms the input data (type: pandas dataframe with
    feature vectors as rows or numpy array) into a nx graph object with
    associated feature vectors and weights. The 2-means clustering
    problem maps to a weighted MaxCut on a complete graph, therefore
    every pair of vertices will be connected by an edge. Only vertex
    weights that cross the cut contribute to the cost.
    Inputs:
    data: type pandas DataFrame or numpy array - the dataset
          to find the 3 means clustering solution on.
    weights: type numpy array - associated coreset weights if
             using a weighted set, default is none.
    '''
    #Generate graph and check if input is DataFrame or numpy array
    if isinstance(data, pd.DataFrame):
        feature_mat = data.to_numpy()
    else:
        feature_mat = data

    (no_vertices, _) = feature_mat.shape
    #Label vertices with even integers for convenience
    vertex_labels = 2 * np.arange(0, no_vertices, 1)
    #Initialise graph and add vertices and edges
    G = nx.Graph()
    G.add_nodes_from(vertex_labels)
    edges = [(2 * i, 2 * j) for i in range(no_vertices) for j in range(i + 1, no_vertices)]
    G.add_edges_from(edges)

    #Add feature vectors to the vertices
    for i in range(no_vertices):
        G.add_node(2 * i, feature_vector=feature_mat[i])

    #Create list of edges to iterate over
    edge_list = list(G.edges)
    #Compute and add edge weights to the graph
    if weights is None:
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
            edge_weight = weights[v_i // 2] * weights[v_j // 2] * dp
            G[v_i][v_j]['weight'] = edge_weight

    return G

def vqe_circuit(G, params, entanglement='linear'):
    '''
    Builds the vqe circuit from the problem graph.
    Inputs:
    G: type networkx graph object - the problem graph
    params: type array - 2 * number of qubit parameters
            for the RY and RX gate (RY params followed by
            RX params.)
    entanglement: linear - one CNOT applied between adjacent
                           qubits (default).
                  full - CNOT applied between all qubits.
    '''
    #Number of qubits is 2x number of data points/vertices
    num_qubits = 2 * len(list(G.nodes))
    #Initialise circuit with equal number of qubits and classical bits
    circuit = QuantumCircuit(num_qubits, num_qubits)
    Rx_index = 0
    Ry_index = int(len(params) / 2)
    while Rx_index <= int((len(params) / 2) - 1):
        #Add RY followed by RX gates on each qubit
        for i in range(num_qubits):
            circuit.ry(params[Rx_index], i)
            circuit.rx(params[Ry_index], i)
            Rx_index +=1
            Ry_index +=1

        circuit.barrier()
        if Rx_index <= int((len(params) / 2) - 2):
            #Add CNOT's to entangle qubits - either linear or full entanglement
            if entanglement == 'linear':
                #Total number of CNOTs in linear entanglement case
                no_cnots = num_qubits - 1
                for i in range(no_cnots):
                    circuit.cnot(i, i + 1)
            if entanglement == 'full':
                #Number of CNOTs with 0 as control in full entanglement case
                no_cnots = num_qubits - 1
                for i in range(no_cnots):
                    for j in range(i + 1, num_qubits):
                        circuit.cnot(i, j)

            circuit.barrier()
        else:
            pass
    #Measure all qubits and write out to classical bits
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit

def draw_circuit(circuit):
    circuit.draw('mpl', style='iqx', fold=50)
    plt.show()


def cost_function(x, G):
    '''
    Evaluates the 3 means cost function on the statevector x.
    Inputs:
    x: statevector - bitstring representing the qubit state
    G: type networkx graph - the problem graph
    '''
    #Start with cost set to 0
    C = 0
    #Create list of edges
    edge_list = list(G.edges)
    for edge in edge_list:
        #Hamiltonian is fourth order so we evaluate the cost
        #function on 4 bits
        v_i = edge[0]
        u_i = edge[0] + 1
        v_j = edge[1]
        u_j = edge[1] + 1
        #Extract binary labels from the bitstring
        y_i = int(x[v_i])
        z_i = int(x[u_i])
        y_j = int(x[v_j])
        z_j = int(x[u_j])
        #Extract the corresponding edge weight
        weight = G[v_i][v_j]['weight']
        #Compute the cost term by term - ft = first term
        ft = (1 - z_i) * ((1 - y_i) * (y_j + z_j - (y_j * z_j)) + y_i * (1 - y_j + (y_j * z_j)))
        st = (z_i * (1 - z_j))
        #Update overall cost
        C += weight * (ft + st)

    return C

def run_vqe(circuit, shots=1024):
    '''
    Runs the vqe circuit and returns the results object
    Inputs:
    circuit: type QuantumCircuit object - the vqe circuit
    shots: type int - number of measurements to be made.
                      (1024 is the default.)
    '''
    qobj = assemble(circuit)
    sim = Aer.get_backend('aer_simulator')
    sim.shots = shots
    result = sim.run(qobj).result()

    return result

def compute_expectation(G, results):
        '''
        Returns the estimated expectation value of a trial state
        |thi(theta)>
        Params:
        G: type graph
        results: type circuit results object
        The results parameter is the object containing
        the results from the measurements of the circuit
        after a number of trials. results.counts() returns
        the dictionary object of states and the number of
        times they were measured.
        '''
        #no_results counts the number of trials to average the
        #sum of the costs.
        no_results = 0
        exp_val = 0
        #Create dictionary
        counts = results.get_counts()
        #Create list of computational basis states
        comp_states = list(counts.keys())
        for state in comp_states:
            #Multiply number of times the state is measured with
            #the cost function evaluated at that state.
            state_count = counts[state] * cost_function(state, G)
            exp_val = exp_val + state_count
            no_results += counts[state]

        return exp_val * (1 / no_results)


def vqe_algorithm(x, G, shots=1024):
    '''
    Performs the vqe algorithm and returns the negative
    computed expected value. We want to optimise the
    parameters to find the maximum expected value, as
    we are using a minimiser as our optimiser we must
    minimise the negative expected value.
    Inputs:
    G: type networkx graph - the problem graph
    x: type array - 2 * number of qubit parameters
            for the RY and RX gate (RY params followed by
            RX params.)
    shots: type int - number of measurements to be made.
                      (1024 is the default.)
    '''
    vqe = vqe_circuit(G, x)
    results = run_vqe(vqe, shots=shots)
    exp_val = compute_expectation(G, results)

    return -exp_val

def optimise_parameters(params, G, shots=1024):
    '''
    Minimises the negative expectation and finds the
    optimal parameters.
    params: type array - 2 * no. qubits for RY and RX
            gate parameters between 0 and 2pi.
    G: type networkx graph - problem graph
    shots: type int - number of measurements to be made.
                      (1024 is the default.)
    '''
    #Make sure even number of params are used.
    no_params = len(params)
    #Constrain params between 0 and 2pi.
    cons1 = tuple({'type': 'ineq', 'fun': lambda x: x[i]} \
                                  for i in range(int(no_params)))
    cons2 = tuple({'type': 'ineq', 'fun': lambda x: (2 * pi) - x[i]} \
                                  for i in range(int(no_params)))
    cons = cons1 + cons2
    #Minimise using COBYLA
    opts = minimize(vqe_algorithm, params, args=(G, shots),
                    method='COBYLA', constraints=cons)

    return opts

def find_optimal_state(G):
    '''
    Finds the optimal state by brute force search
    for a given problem graph G.
    Inputs:
    G: type networkx graph object - problem graph
    '''
    #Length of basis states = 2 * no. vertices of G
    no_vertices = len(list(G.nodes))
    length_basis_states = 2 * no_vertices
    #Generate list of computational basis states.
    comp_states = generate_basis_states(length_basis_states)

    #Start original highest cost at -inf
    best_cost = -np.inf
    best_state = None
    for i, state in enumerate(comp_states):
        if check_clusters(state):
            cost = cost_function(state, G)
            if cost >= best_cost:
                best_cost = cost
                best_state = comp_states[i]
        else:
            pass

    return best_cost, best_state

def check_clusters(state):
    '''
    Counts the number of clusters that a state corresponds to.
    This is used to discount the states with <2 clusters when finding the
    optimal state.
    Inputs:
    state: type string - bitstring state representing the vertices of the graph

    Returns:
    True - if state represents 3 clusters
    False - otherwise
    '''
    #Start with cluster count set to 0
    s1 = 0
    s2 = 0
    s3 = 0
    #Create list of edges
    n = 2
    pairs = [state[i:i+n] for i in range(0, len(state), n)]
    if '00' in pairs:
        s1 = 1
    if '10' in pairs:
        s2 = 1
    if '01' in pairs:
        s3 = 1
    if '11' in pairs:
        s3 = 1
    if (s1 == 1) & (s2 == 1) & (s3 ==1):
        return True
    else:
        return False



def generate_basis_states(n):
    '''
    Returns an ascending order list of the computational
    basis states.
    '''
    W = list(itertools.product(['0','1'], repeat=n))
    comp_states = map(''.join, W)
    return list(comp_states)


def approx_optimal_state(G, depth):
    '''
    #Optimises the vqe parameters to approximate an optimal state
    Inputs:
    G: type networkx graph - the problem graph
    depth: type int - the circuit depth to be used
    '''
    #Each qubit requires two parameters and there are twice as many
    #qubits as nodes in the graph.
    num_qubits = 2 * len(list(G.nodes))
    #Using the formula: num_params = 2 * num_qubits * (depth + 1)
    num_params = 2 * num_qubits * (depth + 1)
    #Initialise random parameters
    params = np.random.uniform(0, 2 * pi, num_params)
    #Find optimal parameters
    opt_params = optimise_parameters(params, G).x
    #Run the circuit with the optimal parameters
    opt_circuit = vqe_circuit(G, opt_params, entanglement='linear')
    #Get the counts from the results
    counts = run_vqe(opt_circuit, shots=1024).get_counts()
    #Find the state that was measured most frequently
    opt_state = max(counts.items(), key=operator.itemgetter(1))[0]

    return opt_state

def approx_partition(G, depth):
    '''
    Finds approximate partition of the data
    '''
    #Simulate VQE to find the aprroximate state
    state = approx_optimal_state(G, depth)
    #Initialise the sets
    s1 = []
    s2 = []
    s3 = []
    #Create list of vertices of G
    vertices = list(G.nodes)
    #Split bitstring into pairs representing vertices
    pairs = [state[i:i+2] for i in range(0, len(state), 2)]
    #Check vertices for which set they correspond to
    for i, vertex in enumerate(pairs):
        if vertex == '00':
            s1.append(vertices[i])
        elif vertex == '10':
            s2.append(vertices[i])
        elif vertex == '01':
            s3.append(vertices[i])
        elif vertex == '11':
            s3.append(vertices[i])

    sets = []
    sets.append(s1)
    sets.append(s2)
    sets.append(s3)

    return sets

def best_partition(G):
    '''
    Maps the best state into the best partition
    '''
    #Initialise the sets
    s1 = []
    s2 = []
    s3 = []
    #Create list of vertices of G
    vertices = list(G.nodes)
    #Brute force the optimal state
    (_, state) = find_optimal_state(G)
    #Split bitstring into pairs representing vertices
    pairs = [state[i:i+2] for i in range(0, len(state), 2)]
    #Check vertices for which set they correspond to
    for i, vertex in enumerate(pairs):
        if vertex == '00':
            s1.append(vertices[i])
        elif vertex == '10':
            s2.append(vertices[i])
        elif vertex == '01':
            s3.append(vertices[i])
        elif vertex == '11':
            s3.append(vertices[i])

    sets = []
    sets.append(s1)
    sets.append(s2)
    sets.append(s3)

    return sets

def approx_clusters(G, data, depth, weights=None):
    '''
    Computes the clusters from the approximate partition.
    '''
    #Compute approx partition
    partition = approx_partition(G, depth)
    #Get length of feature vectors
    cluster_size = len(data[0])
    #Initialise clusters
    c1 = np.zeros(cluster_size)
    c2 = np.zeros(cluster_size)
    c3 = np.zeros(cluster_size)
    clusters = np.array([c1, c2, c3])
    #Compute the sum of weights divided by 3
    if weights is None:
        for i in range(len(partition)):
            for vertex in partition[i]:
                weight = G.nodes[vertex]['feature_vector']
                clusters[i] += weight * (1/len(partition[i]))
    else:
        W = np.sum(weights) / 3
        #Compute cluster centres
        for i in range(len(partition)):
            for vertex in partition[i]:
                weight = weights[int(vertex / 2)] * G.nodes[vertex]['feature_vector']
                clusters[i] += weight * (1/W)

    return clusters


def best_clusters(G, data, weights=None):
    '''
    Computes best clusters from the optimal partition where we have assumed
    equally weighted clusters W1 = W2 = W3 = W/3
    '''
    #Brute force search best partition
    partition = best_partition(G)
    #Get length of feature vectors
    cluster_size = len(data[0])
    #Initialise clusters
    c1 = np.zeros(cluster_size)
    c2 = np.zeros(cluster_size)
    c3 = np.zeros(cluster_size)
    clusters = np.array([c1, c2, c3])
    if weights is None:
        for i in range(len(partition)):
            for vertex in partition[i]:
                weight = G.nodes[vertex]['feature_vector']
                clusters[i] += weight * (1/len(partition[i]))
    else:
        #Compute the sum of weights divided by 3
        W = np.sum(weights) / 3
        #Compute cluster centres
        for i in range(len(partition)):
            for vertex in partition[i]:
                weight = weights[int(vertex / 2)] * G.nodes[vertex]['feature_vector']
                clusters[i] += weight * (1/W)

    return clusters

def cluster_cost_whole_set(data, centres):
    '''
    Computes the cost on the whole data set of the optimal clusters.
    Inputs:
    data: numpy array - feature matrix to evaluate cost against
    '''
    (no_rows, _) = data.shape
    centre1 = centres[0]
    centre2 = centres[1]
    centre3 = centres[2]
    C = 0
    for i in range(no_rows):
        dist = []
        dist.append(np.linalg.norm(data[i] - centre1) ** 2)
        dist.append(np.linalg.norm(data[i] - centre2) ** 2)
        dist.append(np.linalg.norm(data[i] - centre3) ** 2)
        C += min(dist)

    return C

def approximate_n_trials(G, data, coreset, weights, depth, num_runs):
    '''
    Finds best cost on num_runs trials on a coreset.
    '''
    best_cost = np.inf
    best_centres = np.array([0,0,0])
    for i in range(num_runs):
        approx_centres = approx_clusters(G, coreset, depth, weights=weights)
        cost_approx = cluster_cost_whole_set(data, approx_centres)
        if cost_approx < best_cost:
            best_cost = cost_approx
            best_centres = approx_centres

    return (best_cost, best_centres)




def random_n_trials(G, data, sample, num_runs, depth):
    '''
    Finds best clustering cost on num_runs trials on a random sample found
    via simulation of the vqe.
    '''
    approx_cost = np.inf
    for i in range(num_runs):
        approx_centres = approx_clusters(G, sample, depth, weights=None)
        trial_cost = cluster_cost_whole_set(data, approx_centres)
        if trial_cost < approx_cost:
            approx_cost = trial_cost

    return approx_cost
