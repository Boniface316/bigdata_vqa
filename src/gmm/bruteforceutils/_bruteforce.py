import numpy as np
from gmm.helpers import get_weighted_scatter_matrix
from numpy.linalg import inv

# cost for brute-force experiments
def cost_approximate_weighted(coreset, weights, S0, S1):
    w0 = sum(weights[S0])
    w1 = sum(weights[S1])
    W = sum(weights)
    
    w1_div_w0 = 3 - 4*w0/W
    w0_div_w1 = 3 - 4*w1/W

    T_inv = inv(get_weighted_scatter_matrix(coreset, weights))
    cost1 = 0
    for i in S0:
        cost1 += w1_div_w0*weights[i]**2*np.dot(coreset[i], np.dot(T_inv, coreset[i]))
    for i in S1:
        cost1 += w0_div_w1*weights[i]**2*np.dot(coreset[i], np.dot(T_inv, coreset[i]))
    cost2 = 0
    for j in range(len(coreset)):
        for i in range(j):
            if (i in S0 and j in S0):
                cost2 += 2*w1_div_w0*weights[i]*weights[j]*np.dot(coreset[i], np.dot(T_inv, coreset[j]))
            if (i in S1 and j in S1):
                cost2 += 2*w0_div_w1*weights[i]*weights[j]*np.dot(coreset[i], np.dot(T_inv, coreset[j]))
            else:
                cost2 += -2*np.dot(coreset[i], np.dot(T_inv, coreset[j]))

    return -(cost1 + cost2)
