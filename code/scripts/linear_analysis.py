"""
Purpose
-------
This script ####################################################################

It should produce ##############################################################
"""
from __future__ import division, print_function, absolute_import
from matplotlib import colors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.linalg as npl
import os, sys

sys.path.append("code/utils")
from diagnostics import *
from hypothesis import *
from make_class import *
from plot_tool import *


# Create a collection of all subject IDs and all run IDs
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
IDs = list(zip([run_ID for _ in range(16) for run_ID in run_IDs],
               [subject_ID for _ in range(3) for subject_ID in subject_IDs]))
IDs.sort()

# We perform the procedure outlined in this script for each run of each subject:
for ID in IDs:
    run, subject = ID


    # Extract the data of interest
    obj = ds005(subject, run)
    smooth_data = obj.filtered.smooth()
    volume_shape = smooth_data.shape[:3]
    num_volumes = smooth_data.shape[3]

    
    # Load convolution data
    path_convolution = "results/sub%s_run%s/convolution/" % (subject, run)
    conv_gain = np.loadtxt(path_convolution + "conv_gain.txt")
    conv_loss = np.loadtxt(path_convolution + "conv_loss.txt")
    conv_dist2indiff = np.loadtxt(path_convolution + "conv_dist2indiff.txt")


    # Define results directories to which to save the figures produced
    path_result = "results/sub%s_run%s/smoothing/" % (subject, run)
    for path in paths:
        try:
            os.makedirs(path_result)
        except OSError:
            if not os.path.isdir(path_result):
                raise


    # Define linear and quadratic drift components to be used as regressors
    linear_drift = np.linspace(-1, 1, num_volumes)
    quadratic_drift = linear_drift ** 2 - (linear_drift ** 2).mean()

    # Create a design matrix with as many rows as volumes and fill the columns
    # with regressors "parametric gain", "parametric loss", "distance from
    # indifference", the linear drift component, and the quadratic drift
    # component
    design_matrix = np.ones((num_volumes, 6))
    design_matrix[:, 1] = conv_gain
    design_matrix[:, 2] = conv_loss
    design_matrix[:, 3] = conv_dist2indiff
    design_matrix[:, 4] = linear_drift
    design_matrix[:, 5] = quadratic_drift 


    # Compute a threshold for use in identifying
    #to identify the voxels inside the brain
    voxel_means = np.mean(smooth_data, axis=3)
    plt.hist(np.ravel(voxel_means), bins=100)
    plt.savefig(path_result + "voxel_means_hist.png")
    plt.close()
    in_brain_mask = voxel_means > 6000
    #===========================================================================
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

    np.savetxt(path_result + "p_val_gain.txt", np.ravel(p_value_vols[...,0]))
    np.savetxt(path_result + "p_val_loss.txt", np.ravel(p_value_vols[...,1]))
    np.savetxt(path_result + "p_val_dist.txt", np.ravel(p_value_vols[...,2]))
    #===========================================================================
    # plots parameter maps
    mean_vol[~in_brain_mask] = np.nan
    beta_vols[~in_brain_mask] = np.nan

    nice_cmap_values = np.loadtxt("actc.txt")
    nice_cmap = colors.ListedColormap(nice_cmap_values, "actc")

    gain_beta = plot_volume(beta_vols[...,0])
    plt.imshow(gain_beta, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
    plt.colorbar()
    plt.title("Beta estimates for gain")
    plt.savefig(path_result + "parameter_map_gain.png")
    plt.close()

    loss_beta = plot_volume(beta_vols[...,1])
    plt.imshow(loss_beta, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
    plt.colorbar()
    plt.title("Beta estimates for loss")
    plt.savefig(path_result + "parameter_map_loss.png")
    plt.close()

    dist_beta = plot_volume(beta_vols[...,2])
    plt.imshow(dist_beta, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
    plt.colorbar()
    plt.title("Beta estimates for euclidean distance")
    plt.savefig(path_result + "parameter_map_dist.png")
    plt.close()
