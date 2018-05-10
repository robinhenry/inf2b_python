import numpy as np
from scipy.stats import mode
from MySqDist import *

def my_knn_classify(Xtrn, Ctrn, Xtst, Ks):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    #   Ks   : List of the numbers of nearest neighbours in Xtrn
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)
    
    # Matrix sizes
    N = np.shape(Xtst)[0]           # number of test samples
    L = np.shape(Ks)[0]                # number of different k-values to use

    # Compute distances between each test sample and each training sample
    DI = MySqDist(Xtrn, Xtst)

    # Sort the distances between each test sample and all the training samples
    idx = np.argsort(DI)

    # Initialise prediction matrix (N-by-L)
    Cpreds = np.zeros((N,L))

    # Iterate over each value of k from Ks
    for i in range(0, L):
        # Select the indexes corresponding to k nearest neighbours
        k = Ks[i]
        # Add 1 column in case k==1
        k_idx = np.concatenate((idx[:,0:k],np.ones((N,1), dtype=np.int64)), axis=1)  # k_idx = N-by-(k+1)

        # Choose the most frequent class out of the k neighbours, for each sample
        classes = Ctrn[[k_idx],[0]]
        classes.shape = (N, k+1)
        classes = classes[:,0:-1]                   # remove last column
        modes = mode(classes, axis=1)[0]            # compute the modes
        modes.shape = (N)                           # remove an axis
        Cpreds[:,i] = modes

    return Cpreds