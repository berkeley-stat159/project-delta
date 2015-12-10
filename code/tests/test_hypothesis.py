"""
Test functionality of hypothesis module

Tests can be run from the main project directory with:
    nosetests code/tests/test_hypothesis.py
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
from sklearn.linear_model import LogisticRegression
import nibabel as nib
import numpy as np
from nose.tools import assert_equal
from scipy.stats import gamma
import sys, os

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

sys.path.append("code/utils")
from hypothesis import *
from make_class import *

def test_ttest():

    # Make temporary data files
    jarrods_toys = "http://www.jarrodmillman.com/rcsds/_downloads/"
    bold = jarrods_toys + "ds114_sub009_t2r1.nii"
    with open("ds114_sub009_t2r1.nii", "wb") as outfile:
       outfile.write(urlopen(bold).read())
    conv = jarrods_toys + "ds114_sub009_t2r1_conv.txt"
    with open("ds114_sub009_t2r1_conv.txt", "wb") as outfile:
       outfile.write(urlopen(conv).read())

    # Load BOLD and convolved data, excluding first four volumes
    data = nib.load("ds114_sub009_t2r1.nii").get_data()[..., 4:]
    convolved = np.loadtxt("ds114_sub009_t2r1_conv.txt")[4:]

    # Construct design matrix:
    #   Column one is the convolved data, and
    #   Column two is a vetor of ones
    design = np.ones((len(convolved), 2))
    design[:, 0] = convolved

    # Reshape the 4D data to voxel by time 2D
    # Transpose to give time by voxel 2D
    data_2d = np.reshape(data, (-1, data.shape[-1]))
    betas = npl.pinv(design).dot(data_2d.T)

    # Reshape into 4D array
    betas_4d = np.reshape(betas.T, data.shape[:-1] + (-1,))

    # Perform and assess the validity of the t-test
    t1, p1 = ttest(design, betas, data_2d)
    assert np.all(p1 >= 0) 
    assert np.all(p1 <= 1)

    # Delete temporary test files
    os.remove("ds114_sub009_t2r1.nii")
    os.remove("ds114_sub009_t2r1_conv.txt")

def test_waldtest():

    # Load the dummy dataset into the Python environment
    obj = ds005("test", "001")

    # Create necessary input variables
    design_matrix = obj.design_matrix()
    log_model = LogisticRegression().fit(design_matrix, obj.behav[:, 5])
    beta_hat = log_model.coef_.ravel()
    probability_estimates = log_model.predict_proba(design_matrix)

    # Assess proper raising of assertion
    design_matrix_wrong_dim = ds005("test", "001",
                                    rm_nonresp=False).design_matrix()
    assert_raises(AssertionError, waldtest, design_matrix_wrong_dim, beta_hat,
                  probability_estimates)
    
    # Expect none to be statistically significant
    p_values = waldtest(design_matrix, beta_hat, probability_estimates)
    for p_value in p_values: assert p_value > 0.05
