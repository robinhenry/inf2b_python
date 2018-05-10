import numpy as np

def MySqDist(Xtrn, Xtst):
    # Compute square distances between 2 matrix of samples
    # Inputs:
    #   Xtrn: M-by-D matrix of M samples, each of dimension D
    #   Xtst: N-by-D matrix of N samples, each of dimension D
    # Ouptut:
    #   DI: N-by-M euclidean square-distance matrix, where DI(i,j) is the distance
    #       between sample Xtst(i,:) and sample Xtrn(j,:)
    
    # Compute the squared distance, using vectorisation
    XX = np.sum(np.power(Xtst, 2), 1)
    YY = np.sum(np.power(Xtrn, 2), 1)
    DI = 2 * np.dot(Xtst, np.transpose(Xtrn))
    DI = XX[:, np.newaxis] - DI
    DI = DI + (YY[:,np.newaxis]).T

    return DI