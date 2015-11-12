from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.linalg as npl
from scipy.optimize import fmin_bfgs

def read_files(filename):
	"""read from behav.txt and covert data to array format

	Parameter:
	---------
	filename: behav.txt file

	Return:
	------
	val_arr: 
		values read from behav.txt and sorted as an array
	"""
	with open(filename,"r") as infile:
		lines = infile.readlines()
	lines = lines[1:]
	val = [line.split() for line in lines]
	val_arr = np.array(val, dtype=float)
	return val_arr

def construct_mat(array):
	"""Construct the design matrix using the array return from the
	read_files function.

	Parameter:
	----------
	array: 2-D array
		return by read_files

	Return:
	------
	design_matrix: 2-D array 
		The first column is the intercept, the rest columns are features
	"""
	mat = array[:,[1,2,4,5]]
	mat = mat[mat[:,2]!=-1,:]
	diagOfGambleMat = np.array([np.sum(mat[:,2]==1), np.sum(mat[:,2]==3)])
	distance = np.sqrt(np.sum((mat[:,:2]-diagOfGambleMat)**2, axis=1))
	
	design_matrix = np.ones((mat.shape[0],mat.shape[1]+1))
	design_matrix[:,1:] = mat
	design_matrix[:,3] = distance
	return design_matrix

	