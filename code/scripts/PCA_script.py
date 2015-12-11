""" Script of PCA to determine spatial patterns 
For greatest amount of variability in a time seires of images
Find singular value decomposition of the data matrix

Goal:
	1. Find linearly independent sources
	2. Simplify data and filter out unwanted components and can be
		used in the preprocessing stage as a data reduction tool 

Run with:
	python code/scripts PCA_script.py

in the main project directory
"""

import numpy as np
import nibabel as nib
import os
import sys
import pandas as pd
import numpy.linalg as npl
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append("code/utils")
from make_class import *

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

The first principal component vector defines the line that is as close as possible
to the data. 
'''

# Use filtered data 
sub = ds005('001', '001').filtered
# Check the data type
sub.data.dtype
# Convert to float if necessary
data = sub.data.astype(float)
data.dtype

vol_shape, n_trs = sub.data.shape[:-1], sub.data.shape[-1]
data2d = np.reshape(data, (-1, n_trs))
data2d.shape
#(902629, 240)

# Subtract the mean of first dimension 
data_dm = data2d - np.mean(data2d, axis = 0)

# Check if all entries all approximately 0
np.mean(data_dm, axis = 0)
# Compute Covariance Matrix
C = data_dm.T.dot(data_dm)

# Check the dimension of C : time * time
C.shape
# Perform singular value decomposition of C
U, S, VT = npl.svd(C)

'''
A scree plot is a graphical dispaly of the variance of the each component
in the dataset which is used to determine how many components should be retained in 
order to explain a high percentage of the variation in the data. 
'''

# Eigenvalues
eigvals = S**2 / np.cumsum(S)[-1]
fig = plt.figure(figsize=(8,5))

sing_vals = np.arange(n_trs) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Time')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

plt.close()
# Indication of subtracting mean for the second dimension

'''
Subtract means over the second dimension time.
The idea is that out of every possible linear combination of the loadings from the 
	two dimension has squared sum of 1, this particular linear combination yields 
	the highest variance. It is necessary to consider only this linear combinations,
	since otherwise we could increase them arbitrarily in order to blow up the variance.

'''
data_dm_dm = data_dm - np.mean(data_dm, axis=1).reshape((-1, 1))
# Check the resulting numbers are close to zero
data_dm_dm.mean(axis=1)

# Semi-definite postive Matrix 
C_dm_dm = data_dm_dm.T.dot(data_dm_dm)
U, S, VT = npl.svd(C_dm_dm)

# Eigenvalues
eigvals = S**2 / np.cumsum(S)[-1]
fig = plt.figure(figsize=(8,5))

sing_vals = np.arange(n_trs) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Component number')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()
# The plot shows the variance for the first component and then for the subsequent components,
# it shows the additional variance that each component is adding

# Comput the explained variance
# If we projected the observations onto this line, then the resulting projected
# observations would have the largest possible variance; projecting the observations
# onto any other line would yield projected observations with lower variance. 

# The number of Eigenvalues greater than 1 is one, and it 
# explains 25.943916% 
#[ 9.99901225e-01   1.72152618e-05   1.52364491e-05   1.04407335e-05 7.20936171e-06]
pca = PCA(n_components=5)
pca.fit(C)
print(pca.explained_variance_ratio_) 


# Fit the model with X and apply the dimensionality reduction on X
X = pca.fit_transform(C)


  