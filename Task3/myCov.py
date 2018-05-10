import numpy as np 

def myCov(matrix, mu):
    # Input:
    #   matrix : L-by-D data matrix
    #   mu     : D-by-1 sample mean vector
    # Output:
    #   Covs: D-by-D sample covariance matrix.

    # Size
    L = matrix.shape[0]

    # Compute sample covariance matrix
    diff = matrix.T - mu[:, np.newaxis]
    Covs = np.dot(diff, diff.T) / L

    return Covs
