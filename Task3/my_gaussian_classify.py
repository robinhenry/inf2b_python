import numpy as np
from logdet import logdet
from myMean import myMean
from myCov import myCov


def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of label vector for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data matrix (dtype=np.float_)
    #   epsilon   : A scalar parameter for regularisation (type=float)
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=int_)
    #  Ms    : D-by-K ndarray of mean vectors (dtype=np.float_)
    #  Covs  : D-by-D-by-K ndarray of covariance matrices (dtype=np.float_)

    # Size of matrices
    D = Xtrn.shape[1]
    N = Xtst.shape[0]
    K = 26                   # number of classes

    # Compute matrix of sample mean vectors
    #    & 3D array of sample covariance matrices (including regularisation)
    Ms = np.zeros((D, K))
    Covs = np.zeros((D, D, K))
    for k in range(0, K):
        samples = Xtrn[Ctrn == k, :]
        mu = myMean(samples)
        Ms[:, k] = mu
        Covs[:, :, k] = myCov(samples, mu) + np.eye(D) * epsilon

    # NB: No need to include the prior probability to compute the posterior
    #     probability, since we assume a uniform prior distribution over class

    # Compute posterior probabilities for the test samples, in the log domain
    post_log = np.zeros((N, K))
    for k in range(0, K):
        mu = Ms[:,k]
        sigma = Covs[:,:,k]
        diff = Xtst.T - mu
        pro = np.dot( np.dot(diff.T, np.inv(sigma)), diff )
        post_matrix = - 0.5 * pro - 0.5 * logdet(sigma)
        post_log[:,k] = np.diag(post_matrix)

    # Choose the class corresponding to the max posterior probability, for each test sample
    Cpreds = np.argmax(post_log, axis=1)

    return (Cpreds, Ms, Covs)
