""" Script to run linear analysis on FMRI run

This is a revised version of diagnosis_script.py

Requirements:
------------
run "make color" in the command line before run this script!!!!!!

Differences:
-----------
1) Adding linear drift and quadratic drift to linear model
2) Skip the step of detecting outiliers since we have a sense from the
   diagnosis_script that there is rarely outliers. We might add PCA for
   outlier detection.
3) Using smoothed data
4) Plot new functional parameter map image
"""

from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.linalg as npl
from scipy.ndimage import gaussian_filter
from matplotlib import colors

import sys
sys.path.append("code/utils")
from make_class import *
from plot_tool import *
from diagnostics import *
from hypothesis import *

# create a subject
sub = ds005("001","001")
data = sub.filtered.data
affine = sub.filtered.affine
# Smooth by 2 voxel SD in all three spatial dimensions
smooth_data = gaussian_filter(data, [2, 2, 2, 0])
smooth_data = smooth_data[...,4:]
vol_shape, n_trs = smooth_data.shape[:-1], smooth_data.shape[-1]
path = "results_sub001run001/"

#========================================================================================================
# Constructing design matrix
convolved_1 = np.loadtxt(path+'conv001.txt') 
convolved_2 = np.loadtxt(path+'conv002.txt')
convolved_3 = np.loadtxt(path+'conv003.txt')

convolved1 = convolved_1[4:] 
convolved2 = convolved_2[4:]
convolved3 = convolved_3[4:]

N = len(convolved1)
X = np.ones((N, 6)) #make a metrix

X[:, 0] = convolved1 # gain
X[:, 1] = convolved2 # loss
X[:, 2] = convolved3 # dist
linear_drift = np.linspace(-1, 1, n_trs)
X[:, 3] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
X[:, 4] = quadratic_drift 

#========================================================================================================
# find threshold to identify the voxels inside the brain
mean_vol = np.mean(smooth_data, axis=-1)
plt.hist(np.ravel(mean_vol), bins=100)
plt.savefig(path+'mean_vol_hist.png')
plt.close()
in_brain_mask = mean_vol > 6000
#========================================================================================================
# Getting beta hats
Y = smooth_data[in_brain_mask].T
P = 6
Xp = npl.pinv(X)
beta_hat = Xp.dot(Y)
beta_vols = np.zeros(vol_shape + (P,))
beta_vols[in_brain_mask] = beta_hat.T

p_value_vols = np.zeros_like(beta_vols)
p_value_vols[p_value_vols==0] = np.nan
t_score, p_val = t_statistic(X, beta_vols[in_brain_mask].T, Y.T)
p_value_vols[in_brain_mask] = p_val.T

np.savetxt(path+"p_val_gain.txt", np.ravel(p_value_vols[...,0]))
np.savetxt(path+"p_val_loss.txt", np.ravel(p_value_vols[...,1]))
np.savetxt(path+"p_val_dist.txt", np.ravel(p_value_vols[...,2]))
#=========================================================================================================
# plots parameter maps
mean_vol[~in_brain_mask] = np.nan
beta_vols[~in_brain_mask] = np.nan

nice_cmap_values = np.loadtxt('actc.txt')
nice_cmap = colors.ListedColormap(nice_cmap_values, 'actc')

gain_beta_image = plot_volume(beta_vols[...,0])
plt.imshow(gain_beta_image, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("Beta estimates for gain")
plt.savefig(path+'parameter_map_gain.png')
plt.close()

loss_beta_image = plot_volume(beta_vols[...,1])
plt.imshow(loss_beta_image, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("Beta estimates for loss")
plt.savefig(path+'parameter_map_loss.png')
plt.close()

dist_beta_image = plot_volume(beta_vols[...,2])
plt.imshow(dist_beta_image, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("Beta estimates for euclidean distance")
plt.savefig(path+'parameter_map_dist.png')
plt.close()
