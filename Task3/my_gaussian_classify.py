import numpy as np

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

    #YourCode - Bayes classification with multivariate Gaussian distributions.

    return (Cpreds, Ms, Covs)
