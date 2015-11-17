"""
Testing functions in utils_functions.py

Run with:
	nosetests test_utils_functions.py

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import sys
sys.path.append("../.")
from utils_functions import *

filename = "../.././behavdata.txt"
def test_read_txt_files():
	arr = read_txt_files(filename)
	assert isinstance(arr, np.ndarray)
	return True

def test_construct_mat():
	arr = read_txt_files(filename)
	design_mat = construct_mat(arr)
	assert len(design_mat.shape)==2 and design_mat.shape[0] <= arr.shape[0]
	return True