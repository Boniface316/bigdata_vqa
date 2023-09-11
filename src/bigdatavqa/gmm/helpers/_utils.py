import numpy as np
from numpy.linalg import inv

# helper functions
def get_mean(coreset):
    dim = len(coreset[0])
    size_coreset = len(coreset)
    mu = np.zeros(dim)
    for i in range(size_coreset):
        mu += coreset[i]
    return mu/size_coreset
    
def get_weighted_mean(coreset, weights):
    dim = len(coreset[0])
    size_coreset = len(coreset)
    mu = np.zeros(dim)
    for i in range(size_coreset):
        mu += coreset[i]*weights[i]
    return mu/sum(weights)

def get_scatter_matrix(coreset):
    dim = len(coreset[0])
    size_coreset = len(coreset)
    T = np.zeros((dim,dim))
    mu = get_mean(coreset)
    for i in range(size_coreset):
        T += np.outer((coreset[i] - mu),(coreset[i] - mu))
    return T
    
def get_weighted_scatter_matrix(coreset, weights):
    dim = len(coreset[0])
    size_coreset = len(coreset)
    T = np.zeros((dim,dim))
    mu = get_weighted_mean(coreset, weights)
    for i in range(size_coreset):
        T += weights[i]*np.outer((coreset[i] - mu),(coreset[i] - mu))
    return T

def get_matrix_inverse(matrix):
    return inv(matrix)
