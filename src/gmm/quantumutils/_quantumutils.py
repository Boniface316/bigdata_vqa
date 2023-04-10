import numpy as np
from qiskit.opflow.primitive_ops import PauliSumOp
from numpy.linalg import inv
from gmm.helpers import get_weighted_scatter_matrix


# Helper functions for Hamiltonian creation

def Z_i(i, length):
    """ 
    if index i is in the range 0, ..., length-1, the function returns the operator Z_i
    else: the funtion returns the pauli string consisting of pauli I's only
    length is the number of pauli operators tensorised
    """
    pauli_string = ""
    for j in range(length):
        if i == j:
            pauli_string += "Z"
        else:
            pauli_string += "I"
    return pauli_string

def Z_ij(i, j, length):
    pauli_string = ""
    if i == j:
        pauli_string = Z_i(-1, length) # return 'II...II'
    else:
        for k in range(length):
            if k == i or k == j:
                pauli_string += "Z"
            else:
                pauli_string += "I"
    return pauli_string
    
# Create Hamiltonian for our problem
def create_hamiltonian(coreset, weights):

    paulis = []
    pauli_weights = []
    
    T_inv = inv(get_weighted_scatter_matrix(coreset, weights))

    W = sum(weights)

    for i in range(len(coreset)):
        paulis += [Z_i(-1, len(coreset))]
        pauli_weights += [weights[i]**2*np.dot(coreset[i], np.dot(T_inv, coreset[i]))]
    
        for l in range(len(coreset)):
            paulis += [Z_ij(i,l,len(coreset))]
            pauli_weights += [-2*weights[l]*weights[i]**2*np.dot(coreset[i], np.dot(T_inv, coreset[i]))/W]
            
    for j in range(len(coreset)):
        for i in range(j):
            paulis += [Z_ij(i,j,len(coreset))]
            pauli_weights += [2*weights[i]*weights[j]*np.dot(coreset[i], np.dot(T_inv, coreset[j]))]
            for l in range(len(coreset)):
                paulis += [Z_ij(i,l,len(coreset))]
                pauli_weights += [-2*weights[l]*weights[i]*weights[j]*np.dot(coreset[i], np.dot(T_inv, coreset[j]))/W]
                paulis += [Z_ij(j,l,len(coreset))]
                pauli_weights += [-2*weights[l]*weights[i]*weights[j]*np.dot(coreset[i], np.dot(T_inv, coreset[j]))/W]
            
            
    pauli_op = [([pauli,weight]) for pauli,weight in zip(paulis,pauli_weights)]
    hamiltonian = PauliSumOp.from_list([ op for op in pauli_op])
    # we consider the negative of the hamiltonian since VQE approximates the minimum (and not the maximum)
    hamiltonian = -hamiltonian 
    
    return hamiltonian
