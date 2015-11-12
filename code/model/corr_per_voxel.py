from __future__ import division, print_function, absolute_import
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from stimuli import events2neural

def corr(nii_file, cond_file):
	"""Find the correlations between time course and each voxel
	
	Parameters:
	----------
	nii_file: bold.nii file 
	cond_file: condition file

	Return:
	-------
	correlations: an array (n_voxels, )
	"""
	img = nib.load(nii_file)
	n_trs = img.shape[-1]
	TR = 2 #The TR (time between scans) is 2 seconds
	time_course = events2neural(cond_file, 2, n_trs)
	# Call the events2neural function to generate the on-off values for each volume
	data = img.get_data() 
	# Using slicing, drop the first 4 volumes, and the first 4 on-off values
	data = data[..., 4:]
	time_course = time_course[4:]
	n_voxels = np.prod(data.shape[:-1]) # Calculate the number of voxels (number of elements in one volume)
	data_2d = np.reshape(data, (n_voxels, data.shape[-1])) # Reshape 4D array to 2D array n_voxels by n_volumes
	correlations_1d = np.zeros((n_voxels,)) # Make a 1D array of size (n_voxels,) to hold the correlation values
	for i in range(n_voxels): # Loop over voxels filling in correlation at this voxel
    	correlations_1d[i] = np.corrcoef(time_course, data_2d[i, :])[0, 1]
	correlations = np.reshape(correlations_1d, data.shape[:-1]) # Reshape the correlations array back to 3D
	plt.imshow(correlations[:, :, 14]) # Plot the middle slice of the third axis from the correlations array
	return correlations #get correlations of two value



