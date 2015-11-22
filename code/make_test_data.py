from __future__ import absolute_import, division, print_function
import numpy as np
import os

# Paths to directories containing the test subject's data 
path_data = "../data/ds005/testsub/"
path_BOLD = path_data + "BOLD/task001_testrun/bold.nii.gz"
path_behav = path_data + "behav/task001_testrun/behavdata.txt"

# Create these directories
os.makedirs(path_BOLD)
os.makedirs(path_behav)

# Give the BOLD data the identity affine for simplicity
affine = np.eye(4)

# The fMRI data consists of three volumes of shape (3, 3, 3)
# Corner elements increase by 1 per unit time
# Edge elements increase by 2 per unit time
# Center of face elements increase by 3 per unit time
# The center element increases by 4 per unit time
data = np.array([[[[ 0,  1,  2],
                   [ 1,  3,  5],
                   [ 0,  1,  2]],

                  [[ 1,  3,  5],
                   [ 2,  5,  8],
                   [ 1,  3,  5]],

                  [[ 0,  1,  2],
                   [ 1,  3,  5],
                   [ 0,  1,  2]]],


                 [[[ 1,  3,  5],
                   [ 2,  5,  8],
                   [ 1,  3,  5]],

                  [[ 2,  5,  8],
                   [ 3,  7, 11],
                   [ 2,  5,  8]],

                  [[ 1,  3,  5],
                   [ 2,  5,  8],
                   [ 1,  3,  5]]],


                 [[[ 0,  1,  2],
                   [ 1,  3,  5],
                   [ 0,  1,  2]],

                  [[ 1,  3,  5],
                   [ 2,  5,  8],
                   [ 1,  3,  5]],

                  [[ 0,  1,  2],
                   [ 1,  3,  5],
                   [ 0,  1,  2]]]])

# BOLD.nii contains the above two elements
BOLD = nib.nift1.NiftiImage(data, affine)
nib.save(BOLD, path_BOLD)

# The behavioral data consists of four rows: a row of headers, and one row for
# each of three trials that occur at times 0.0, 2.0, and 4.0
behav = "onset\tgain\tloss\tPTval\trespnum\trespcat\tRT\n"
behav = behav + "0.00\t10\t20\t-9.80\t4\t0\t1.077\n"
behav = behav + "2.00\t20\t20\t0.20\t0\t-1\t0.000\n"
behav = behav + "4.00\t30\t20\t10.20\t2\t1\t1.328"

# Create behavdata.txt and open to write
f = open(path_behav + "behavdata.txt")
f.write(behav)
f.close()