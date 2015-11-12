from __future__ import division, print_function, absolute_import
import numpy as np

def read_files(filename):
	"""read from behav.txt and covert data to array format

	Parameter:
	---------
	filename: behavdata.txt file

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

def construct_mat(array, includeED=True):
	"""Construct the design matrix using the array return from the
	read_files function.

	Parameter:
	----------
	array: 2-D array
		return by read_files
	includeED: boolean (True or False)
		True (default): include Euclidean distance as the 3rd regressor (4th column of design_matrix)
		False: not include. Only two regressors (gain and loss) are included.

	Return:
	------
	design_matrix: 2-D array 
		The first column is always intercept, and the last column is the response (0 and 1)
		The column names are (intercept, gain, loss, distance(if includeED=True), response) 
	"""
	array = array[array[:,-2]!=-1,:]
	if includeED:
		mat = array[:,[1,2,4,5]]
		diagOfGambleMat = np.array([np.sum(mat[:,2]==1), np.sum(mat[:,2]==3)])
		distance = np.sqrt(np.sum((mat[:,:2]-diagOfGambleMat)**2, axis=1))
		design_matrix = np.ones((mat.shape[0],mat.shape[1]+1))
		design_matrix[:,1:] = mat
		design_matrix[:,3] = distance
	else:
		mat = array[:,[1,2,5]]
		design_matrix = np.ones((mat.shape[0],mat.shape[1]+1))
		design_matrix[:,1:] = mat
	return design_matrix

	