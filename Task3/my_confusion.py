#
# A sample template for my_confusion.py
#
# Note that:
#   We assume that the original labels have been pre-processed so that
#   class number starts at 0 rather than 1 to meet the NumPy's array indexing
#   policy. For example, if the number of classes is K, label values are in
#   [0,K-1] range. (This conversion does not apply to codig wih Matlab)

import numpy as np

def my_confusion(Ctrues, Cpreds):
    # Input:
    #   Ctrues : N-by-1 ndarray of ground truth label vector (dtype=np.int_)
    #   Cpreds : N-by-1 ndarray of predicted label vector (dtype=np.int_)
    # Output:
    #   CM : K-by-K ndarray of confusion matrix, where CM[i,j] is the number of samples whose target is the ith class that was classified as j (dtype=np.int_)
    #   acc : accuracy (i.e. correct classification rate) (type=float)
    #

    return (CM, acc)
