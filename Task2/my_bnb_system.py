# A sample template for my_bnb_system.py

import numpy as np
import time
from scipy.io import loadmat
from my_confusion import my_confusion
from my_bnb_classify import my_bnb_classify
from my_bnb_classify_noLog_smoothing import my_bnb_classify_noLog_smoothing


# Path
path = '/Users/robin/Projects/python/inf2B_cw2/'

# Load the data set
filename = path + "data.mat"
data = loadmat(filename)

# Feature vectors: Convert uint8 to double   (but do not divide by 255)
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_)
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float_)
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

# Prepare measuring time
time.clock()

# Run classification
threshold = 1.0
Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
Cpreds1 = my_bnb_classify_noLog_smoothing(Xtrn, Ctrn, Xtst, threshold)

# Measure the user time taken, and display it.
elapsed_time = time.clock()

# Get a confusion matrix and accuracy
(CM, acc) = my_confusion(Ctst, Cpreds)
(CM1, acc1) = my_confusion(Ctst, Cpreds1)

# Save the confusion matrix as "Task2/cm.mat".
np.save(path + 'Task2/cm.mat', CM)

# Display the required information - N, Nerrs, acc.
N = Ctst.shape[0]
Nerrs = int(N * (1-acc))
Nerrs1 = int(N * (1-acc1))

print(' Num. of test samples: ' + str(N) +
          ',\t Num. of errors: ' + str(Nerrs) + ',\t Accuracy: {0:.3f}.' .format(acc))
print(' Num. of test samples: ' + str(N) +
          ',\t Num. of errors: ' + str(Nerrs1) + ',\t Accuracy: {0:.3f}.' .format(acc1))