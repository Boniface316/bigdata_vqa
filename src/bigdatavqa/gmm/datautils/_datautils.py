import numpy as np
from scipy.stats import multivariate_normal



def create_dataset():

    random_seed=1000
    
    # List of covariance values
    cov_val = [-0.8, -0.8]
    
    # Setting means
    mean = np.array([[0,0], [7,1]])
    
    X = np.zeros((1000,2))
    
    # Iterating over different covariance values
    for idx, val in enumerate(cov_val):
    
        # Initializing the covariance matrix
        cov = np.array([[1, val], [val, 1]])
	     
	# Generating a Gaussian bivariate distribution
	# with given mean and covariance matrix
	distr = multivariate_normal(cov = cov, mean = mean[idx],
		                        seed = random_seed)
	     
	# Generating 500 samples out of the
	# distribution
	data = distr.rvs(size = 500)
	     
	X[500*idx:500*(idx+1)][:] = data
	    
    return X
