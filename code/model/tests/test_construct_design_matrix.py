"""
Testing the functions in the module construct_design_matrix
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from construct_design_matrix import *

def test_read_files():
	filename = ".././behavdata.txt"
	arr = read_files(filename)
	assert isinstance(arr, np.ndarray)

def test_construct_mat():
	filename = ".././behavdata.txt"
	arr = read_files(filename)
	design_mat = construct_mat(arr)
	assert len(design_mat.shape)==2 and design_mat.shape[0] <= arr.shape[0]