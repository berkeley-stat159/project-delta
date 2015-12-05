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
import sys

sys.path.append("code/utils")
from make_class import *

from hypothesis import * 
from convolution import *

# Path to retrieve image data used in class
path_data = 'ds114/'


def test_hypothesis():
	
	img = nib.load(path_data + 'ds114_sub009_t2r1.nii')
	data = img.get_data()
	data = data[...,4:]

	# Retrieve convolved data from the same path
	convolved = np.loadtxt(path_data + 'ds114_sub009_t2r1_conv.txt')[4:]

	# Constructing design matrix,
	# First column is convovled regressor, Second column all ones
	design = np.ones((len(convolved), 2))
	design[:, 0] = convolved 

	# Reshape the 4D data to voxel by time 2D
	# Transpose to give time by voxel 2D
	data_2d = np.reshape(data, (-1, data.shape[-1]))
	betas = npl.pinv(design).dot(data_2d.T)

	# Reshape into 4D array
	betas_4d = np.reshape(betas.T, img.shape[:-1] + (-1,))

	#Change t_test() to t_statistic() to pass nosetests
	t1, p1 = t_statistic(design, betas, data_2d)

	assert np.all(p1 >= 0) 
	assert np.all( p1 <= 1)

