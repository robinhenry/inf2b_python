# my_knn_system.py script

import numpy as np
from scipy.io import loadmat
import time
from my_knn_classify import my_knn_classify
from my_confusion import my_confusion

# Path
path = '/Users/robin/Projects/python/inf2B_cw2/'

# Load the data set
filename = path + "data.mat"
data = loadmat(filename)

# Feature vectors: Convert uint8 to double, and divide by 255.
Xtrn = data['dataset']['train'][0, 0]['images'][0,
                                                0].astype(dtype=np.float_) / 255.0
Xtst = data['dataset']['test'][0, 0]['images'][0,
                                               0].astype(dtype=np.float_) / 255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0, 0]['labels'][0,
                                                0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0, 0]['labels'][0,
                                               0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

# Prepare measuring time
time.clock()

# Run K-NN classification
kb = [1, 3, 5, 10, 20]
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb)

# Measure the user time taken, and display it.
elapsed_time = time.clock()
print('\nTime taken by my_knn_classify(): {0:.3f} seconds.\n\n' .format(
    elapsed_time))

# for each k in kb:
for i in range(0, len(kb)):
    # Get confusion matrix and accuracy
    (CM, acc) = my_confusion(Ctst, Cpreds[:, i])
    # Save each confusion matrix
    file = path + 'Task1/cm' + str(kb[i]) + '.mat'
    np.save(file, CM)
    # Display the required information - k, N, Nerrs, acc
    N = Ctst.shape[0]
    Nerrs = int(N * (1-acc))
    print('k: ' + str(kb[i]) + ',\t Num. of test samples: ' + str(N) +
          ',\t Num. of errors: ' + str(Nerrs) + ',\t Accuracy: {0:.3f}.' .format(acc))
