"""
==================Test file for hypothesis.py======================

Test T test for multiple linear regression model

Run with:
        nosetests code/tests/test_hypothesis.py
in the main project directory
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import nibabel as nib
from nose.tools import assert_equal
from scipy.stats import gamma
import sys, os

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

sys.path.append("code/utils")
from hypothesis import * 
from convolution import *

def test_hypothesis():

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

    # Change t_test() to t_statistic() to pass nosetests
    t1, p1 = t_statistic(design, betas, data_2d)

    assert np.all(p1 >= 0) 
    assert np.all(p1 <= 1)

    # Delete temporary test files
    os.remove("ds114_sub009_t2r1.nii")
    os.remove("ds114_sub009_t2r1_conv.txt")