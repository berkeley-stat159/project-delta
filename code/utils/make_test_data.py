from __future__ import absolute_import, division, print_function
import nibabel as nib
import numpy as np
import os

# Paths to directories containing the test subjects' data 
path_sub1, path_sub2 = "data/ds005/subtest1/", "data/ds005/subtest2/"
path_BOLD_11 = path_sub1 + "BOLD/task001_run001/"
path_BOLD_12 = path_sub1 + "BOLD/task001_run002/"
path_BOLD_21 = path_sub2 + "BOLD/task001_run001/"
path_BOLD_22 = path_sub2 + "BOLD/task001_run002/"
path_behav_11 = path_sub1 + "behav/task001_run001/"
path_behav_12 = path_sub1 + "behav/task001_run002/"
path_behav_21 = path_sub2 + "behav/task001_run001/"
path_behav_22 = path_sub2 + "behav/task001_run002/"
paths = [path_BOLD_11, path_BOLD_12, path_BOLD_21, path_BOLD_22,
         path_behav_11, path_behav_12, path_behav_21, path_behav_22]

# Create these directories from scratch
for path in paths:
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

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
BOLD = nib.Nifti1Image(data, affine)
for path_BOLD in paths[:4]: nib.save(BOLD, path_BOLD + "bold.nii.gz")

# The behavioral data consists of four rows: a row of headers, and one row for
# each of three trials that occur at times 0.0, 2.0, and 4.0
behav = "onset\tgain\tloss\tPTval\trespnum\trespcat\tRT\n"
behav = behav + "0.00\t10\t20\t-9.80\t4\t0\t1.077\n"
behav = behav + "2.00\t20\t20\t0.20\t0\t-1\t0.000\n"
behav = behav + "4.00\t30\t20\t10.20\t2\t1\t1.328"

# Create behavdata.txt and open to write
for path_behav in paths[4:]:
    f = open(path_behav + "behavdata.txt", "wt")
    f.write(behav)
    f.close()