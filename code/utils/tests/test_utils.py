"""
Testing functions in utils_functions.py

Run with:
	nosetests test_utils.py

"""
from __future__ import division, print_function, absolute_import
import numpy as np
from .. import utils_functions

filename = "behavdata.txt"
def test_read_txt_files():
	arr = utils_functions.read_txt_files(filename)
	assert isinstance(arr, np.ndarray)
	return True

def test_construct_mat():
	arr = utils_functions.read_txt_files(filename)
	design_mat = utils_functions.construct_mat(arr)
	assert len(design_mat.shape)==2 and design_mat.shape[0] <= arr.shape[0]
	return True
