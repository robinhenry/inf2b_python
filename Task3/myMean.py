import numpy as np

def myMean(matrix):
    # Input:
    #   matrix: L-by-D data matrix
    # Output:
    #   mu: D-by-1 column vector of sample mean values, where mu(i) = mean(matrix(:,i)).

    # Check if the matrix is not empty to make sure we do not divide by 0.
    if matrix.shape[0] == 0:
        s = 1
    else: 
        s = matrix.shape[0]
    
    # Compute sample mean vector
    mu = (np.sum(matrix, axis=0) / s).T 

    return mu
