# Import numerical and plotting libraries
import numpy as np
import itertools 
from scipy.ndimage.filters import gaussian_filter

def smooth(input, sigma, slice):
	'''
	Return array of same shape as input.
	The multidimensional filter is implemented as a sequence of one-dimensional
	filters.
	The intermediate arrays are sorted in the same data type as the output.

	Parameters
	----------
	input: A numpy array of an image data in one task run

	sigma: The standard deviations for Gaussian kernel.

	slice: The index time of the fourth dimension.

	Returns
	-------
	Returned array of same shape as input.

	'''
	input_slice = input[...,slice]
	data_array = scipy.ndimage.filters.gaussian_filter(input, sigma)
	return data_array