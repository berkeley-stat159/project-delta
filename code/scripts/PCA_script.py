""" Script of PCA to determine spatial patterns 
For greatest amount of variability in a time seires of images
Find singular value decomposition of the data matrix

Goal:
	1. Find linearly independent sources
	2. Simplify data and filter out unwanted components and can be
		used in the preprocessing stage as a data reduction tool 

Run with:
	python PCA_script.py

in the main project directory
"""

import numpy as np
import nibabel as nib
import os
import sys
import pandas as pd
import numpy.linalg as npl
import matplotlib.pyplot as plt


'''
PCA allows one to determine the spatial patterns that account for the greatest amount of 
variability in a time series of images. 
Find singular value decomposition of the data matrix. 

Usefulness: 1. This decomposition can potentially reveal the nature of the observed signal by 
				finding its linearly independent sources. 
			2. decomposing the signal and ordering the components according to their 
				weight is a useful way to simplify the data or filter our unwanted components, 
				and can be used in the preprocessing stage as a data reduction tool

Capture the most prominent variations across the set of voxels. 
The components may either refelct signals of interest or they may be dominated by aritfacts. 
Assume all variability results from signal, as noise is not included in the model formulation 
'''
#Use filtered data 
sub = run('001', '001', filtered_data=True)
#check the data type
sub.data.dtype
#Convert to float if necessary
data = sub.data.astype(float)
data.dtype

vol_shape, n_trs = sub.data.shape[:-1], sub.data.shape[-1]
data2d = np.reshape(data, (-1, n_trs))

data_dm = data2d - np.mean(data2d, axis = 0)

#check if all entries all approximately 0
np.mean(data_dm, axis = 0)
C = data_dm.T.dot(data_dm)

#Check the dimension of C : time * time
C.shape
U, S, VT = npl.svd(C)

#Eigenvalue
eigvals = S**2 / np.cumsum(S)[-1]
fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(n_trs) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')


plt.plot(S)

# Subtract means over the second dimension time
data_dm_dm = data_dm - np.mean(data_dm, axis=1).reshape((-1, 1))
data_dm_dm.mean(axis=1)
C_dm_d = data_dm_dm.T.dot(data_dm_dm)
U2, S2, VT2 = np.linalg.svd(C_dm_dm)
plt.plot(S2)

# Comput the explained variance
exp_var = S2 / np.sum(S2)
var_sums = np.cumsum(exp_var)

plt.plot(exp_var[np.arange(1,100)], 'b-o')
plt.xlabel("Principal Components")
plt.title("Proportion of Variance Explained by Each Component")


plt.plot(var_sums[np.arange(1,240)], 'b-o')
plt.xlabel("NUmber of Principal Components")
plt.title("Proportion of Variance Explained by Each Component")

  