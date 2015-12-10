"""
This script creates test data to be used to test code in the code/model/ and
code/utils/ directories. This script should be run before checking hashes.
"""
from __future__ import absolute_import, division, print_function
import nibabel as nib
import numpy as np
import os


# Paths to directories containing the test subjects' data 
path_data = "data/ds005/subtest/"
path_BOLD = path_data + "BOLD/task001_run001/"
path_mni = path_data + "model/model001/task001_run001.feat/"
path_behav = path_data + "behav/task001_run001/"
path_cond = path_data + "model/model001/onsets/task001_run001/"
paths = [path_behav, path_BOLD, path_cond, path_mni]


# Create these directories from scratch
for path in paths:
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise # pragma: no cover


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
nib.save(BOLD, path_BOLD + "bold.nii.gz")
nib.save(BOLD, path_mni + "filtered_func_data_mni.nii.gz")


# The behavioral data consists of four rows: a row of headers, and one row for
# each of three trials that occur at times 0, 2, and 4
behav = "onset\tgain\tloss\tPTval\trespnum\trespcat\tRT\n"
behav = behav + "0.00\t10\t20\t-9.80\t4\t0\t1.077\n"
behav = behav + "2.00\t20\t20\t0.20\t0\t-1\t0.000\n"
behav = behav + "4.00\t30\t20\t10.20\t2\t1\t1.328\n"

# Create behavdata.txt and open to write
f = open(path_behav + "behavdata.txt", "wt")
f.write(behav)
f.close()


# The task condition data consists of three important subsets: parametric gain,
# parametric loss, and distance from indifference. Each subset further contains:
#   three rows: one for each of three trials occurring at times 0, 2, and 4
#   three columns: one for each of the onset time, the duration, and parameter
# We begin with the parametric gain:
cond_2 = "0.0000\t1\t-1.0000\n"
cond_2 = cond_2 + "2.0000\t1\t0.0000\n"
cond_2 = cond_2 + "4.0000\t1\t1.0000\n"

f = open(path_cond + "cond002.txt", "wt")
f.write(cond_2)
f.close()


# Now the parametric loss:
cond_3 = "0.0000\t1\t0.0000\n"
cond_3 = cond_3 + "2.0000\t1\t0.0000\n"
cond_3 = cond_3 + "4.0000\t1\t0.0000\n"

f = open(path_cond + "cond003.txt", "wt")
f.write(cond_3)
f.close()


# And finally the distance from indifference:
cond_4 = "0.0000\t1\t-1.0000\n"
cond_4 = cond_4 + "2.0000\t1\t0.0000\n"
cond_4 = cond_4 + "4.0000\t1\t1.0000\n"

f = open(path_cond + "cond004.txt", "wt")
f.write(cond_4)
f.close()
