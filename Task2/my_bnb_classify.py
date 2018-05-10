import numpy as np

def my_bnb_classify(Xtrn, Ctrn, Xtst, threshold):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   threshold   : A scalar threshold (type=float)
    # Output:
    #  Cpreds : N-by-1 ndarray of predicted labels for Xtst (dtype=np.int_)

    # Matrix sizes
    N = Xtst.shape[0]                   # number of test samples
    D = Xtst.shape[1]                   # size of a feature vector
    C = 26                              # number of classes

    # Binarisation of Xtrn and Xtst
    Xtrn_bin = Xtrn >= threshold
    Xtst_bin = Xtst >= threshold

    # Matrices to store characteristics of each class
    trn_zeros = np.zeros((C, D))          # C-by-D matrix, numbers of 0's for each pixel
    trn_ones = np.zeros((C, D))           # C-by_D matrix, numbers of 1's for each pixel

    # Initialise matrix to store likelihoods
    likelihoods = np.zeros((N, C))  

    for k in range(0, C):
        # Select training samples from class k and 1 row in case there is only 1 training sample (just in case)
        Xtrn_k = np.concatenate((Xtrn_bin[(Ctrn == k)[:,0], :], np.zeros((1,D))))
        # Number of samples of class k
        nbr_samples_k = Xtrn_k.shape[0]
        nbr_samples_k = Xtrn_k.shape[0] 
        # P(D_i = 0|C_k), where D_i is the ith element of a feature vector D
        trn_zeros[k,:] = np.sum(Xtrn_k == 0, axis=0) / nbr_samples_k

        # Compute the likelihoods
        factor_0 = np.power(trn_zeros[k,:], (1 - Xtst_bin))
        factor_1 = np.power(trn_ones[k,:], Xtst_bin)

        likelihoods[:,k] = np.prod(factor_0 * factor_1, 1) 

    # NB: No need to multiply the likelihoods by the prior probability, 
    #     since we assume a uniform prior distribution over class

    # Get the maximum posterior probability and find Cpreds
    Cpreds = np.argmax(likelihoods, axis=1)

    return Cpreds