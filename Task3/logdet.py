import numpy as np

def logdet(covariance):
    """
    This should be equivalent to the following:
    
    >>> covariance_logdet = numpy.linalg.slogdet(covariance)[1]
    
    so feel free to use either
    """
    L = np.linalg.cholesky(covariance)
    covariance_logdet = 2*np.sum(np.log(np.diagonal(L)))
    return covariance_logdet
