# my_gaussian_system.py script

import numpy as np
import time
from scipy.io import loadmat
from my_gaussian_classify import my_gaussian_classify
from my_confusion import my_confusion

# Path
path = '/Users/robin/Projects/python/inf2B_cw2/'

# Load the data set
filename = path + "data.mat"
data = loadmat(filename)

# Feature vectors: Convert uint8 to double, and divide by 255
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

# Prepare measuring time
time.clock()

# Run classification
epsilon = 0.01
(Cpreds, Mu, Covs) = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)

# Measure the user time taken, and display it.
elapsed_time = time.clock()

# Get a confusion matrix and accuracy
(CM, acc) = my_confusion(Ctst, Cpreds)

# Save the confusion matrix as "Task3/cm.mat".
np.save(path + 'Task3/cm.mat', CM)

# Save the mean vector and covariance matrix for class 26,
#           i.e. save Mu(:,25) and Cov(:,:,25) as "Task3/m26.mat" and
#           "Task3/cov26.mat", respectively.
np.save(path + 'Task3/m26.mat', Mu[:,25])
np.save(path + 'Task3/cov26.mat', Covs[:,:,25])

# Display the required information - N, Nerrs, acc.
N = Ctst.shape[0]
Nerrs = int(N * (1 - acc))
print(' Num. of test samples: ' + str(N) +
          ',\t Num. of errors: ' + str(Nerrs) + ',\t Accuracy: {0:.3f}.' .format(acc))