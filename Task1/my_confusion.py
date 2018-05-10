import numpy as np

def my_confusion(Ctrues, Cpreds):
    # Input:
    #   Ctrues : N-by-1 ndarray of ground truth label vector (dtype=np.int_)
    #   Cpreds : N-by-1 ndarray of predicted label vector (dtype=np.int_)
    # Output:
    #   CM : K-by-K ndarray of confusion matrix, where CM[i,j] is the number of samples whose target is the ith class that was classified as j (dtype=np.int_)
    #   acc : accuracy (i.e. correct classification rate) (type=float)

    # Number of samples
    N = Cpreds.shape[0]

    # Initialisation of confusion matrix
    K = 26
    CM = np.zeros((K, K))

    # Iterate over each class
    for k in range(0,K):
        # Compute vector of predictions corresponding to truth of class k
        select = Ctrues == k
        select.shape = N
        preds = Cpreds[select].astype(dtype=np.int64)
        # Increment the kth row (samples that should be of class k) in CM
        for j in preds:
            CM[k,j] = CM[k,j] + 1
    
    # Compute accuracy
    acc = np.trace(CM) / Ctrues.shape[0]

    return (CM, acc)