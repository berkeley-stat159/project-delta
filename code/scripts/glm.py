"""
Purpose
-------
This script contains code to fit and assess the use of generalized linear model
on the smoothed data. This analysis cannot be performed before `convolution` and
also follows the logic obtained from `diagnosis` in that it refuses to drop
outlier volumes.

It should produce a total of three figures and six .nii files per run:
- `p_value_dist2indiff.nii.gz`
- `p_value_gain.nii.gz`
- `p_value_loss.nii.gz`
- `regr_coef_by_voxel_dist2indiff.png`
- `regr_coef_by_voxel_gain.png`
- `regr_coef_by_voxel_loss.png`
- `t_stat_dist2indiff.nii.gz`
- `t_stat_gain.nii.gz`
- `t_stat_loss.nii.gz`
It also saves all relevant neural loss aversion data to a single plaintext file
`results/neural_loss_aversion.txt`.
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
from stat_utils import *


# Create a file to use as a repository for neural loss aversion data
with open("results/neural_loss_aversion.txt", "w") as outfile:
    outfile.write("run\tsubject\tneural_loss_aversion\n")


# Create a collection of all subject IDs and all run IDs
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
IDs = list(zip([run_ID for _ in range(16) for run_ID in run_IDs],
               [subject_ID for _ in range(3) for subject_ID in subject_IDs]))
IDs.sort()

# We perform the procedure outlined in this script for each run of each subject:
for ID in IDs:
    run, subject = ID


    # Extract the data of interest
    obj = ds005(subject, run)
    smoothed_data = obj.filtered.smooth()
    affine = obj.filtered.affine
    volume_shape = smoothed_data.shape[:3]
    num_volumes = smoothed_data.shape[3]

    
    # Load convolution data
    path_convolution = "results/run%s/convolution/sub%s/" % ID
    conv_gain = np.loadtxt(path_convolution + "conv_gain.txt")
    conv_loss = np.loadtxt(path_convolution + "conv_loss.txt")
    conv_dist2indiff = np.loadtxt(path_convolution + "conv_dist2indiff.txt")


    # Define results directories to which to save the figures produced
    path_result = "results/run%s/glm/sub%s/" % ID
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
    num_regressors = 6 # Consider one more for the intercept
    design_matrix = np.ones((num_volumes, num_regressors))
    design_matrix[:, 1] = conv_gain
    design_matrix[:, 2] = conv_loss
    design_matrix[:, 3] = conv_dist2indiff
    design_matrix[:, 4] = linear_drift
    design_matrix[:, 5] = quadratic_drift 


    # Compute a threshold for identifying which voxels are inside the brain
    voxel_means = np.mean(smoothed_data, axis=3)
    voxels_in_brain = voxel_means > np.percentile(voxel_means, 80)

    
    # Compute regression coefficients for all voxels over time
    response = smoothed_data[voxels_in_brain].T
    regr_coef, df, MRSS = glm_util(design_matrix, response)
    regr_coef_by_voxel = np.zeros(volume_shape + (num_regressors,))
    regr_coef_by_voxel[voxels_in_brain] = regr_coef.T

    # Assess statistical significance of regressors by voxel
    t_stat_by_voxel = np.zeros(regr_coef_by_voxel.shape)
    p_value_by_voxel = np.zeros(regr_coef_by_voxel.shape)
    t_stat, p_value = ttest(design_matrix, regr_coef, response.T)
    t_stat_by_voxel[voxels_in_brain] = t_stat.T
    p_value_by_voxel[voxels_in_brain] = p_value.T


    # Save the t-statistics as .nii files
    t_stat_gain = nib.Nifti1Image(t_stat_by_voxel[..., 1], affine)
    t_stat_loss = nib.Nifti1Image(t_stat_by_voxel[..., 2], affine)
    t_stat_dist2indiff = nib.Nifti1Image(t_stat_by_voxel[..., 3], affine)

    nib.save(t_stat_gain, path_result + "t_stat_gain.nii.gz")
    nib.save(t_stat_loss, path_result + "t_stat_loss.nii.gz")
    nib.save(t_stat_dist2indiff, path_result + "t_stat_dist2indiff.nii.gz")


    # Save the p-values as .nii files
    p_value_gain = nib.Nifti1Image(p_value_by_voxel[..., 1], affine)
    p_value_loss = nib.Nifti1Image(p_value_by_voxel[..., 2], affine)
    p_value_dist2indiff = nib.Nifti1Image(p_value_by_voxel[..., 3], affine)

    nib.save(p_value_gain, path_result + "p_value_gain.nii.gz")
    nib.save(p_value_loss, path_result + "p_value_loss.nii.gz")
    nib.save(p_value_dist2indiff, path_result + "p_value_dist2indiff.nii.gz")


    # Set up our color utilities
    voxel_means[~voxels_in_brain] = np.nan
    regr_coef_by_voxel[~voxels_in_brain] = np.nan
    nice_cmap_values = np.loadtxt("code/scripts/actc.txt")
    nice_cmap = colors.ListedColormap(nice_cmap_values, "actc")


    # Produce and save a plot of the regression coefficients by voxel for each
    # of the three regressors
    gain_coef_by_voxel = regr_coef_by_voxel[..., 1]
    plot_gain = plot_volume(gain_coef_by_voxel, backdrop=np.nan)
    plt.imshow(plot_gain, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
    plt.colorbar()
    plt.title("Regression Coefficients by Voxel (Gain)")
    plt.savefig(path_result + "regr_coef_by_voxel_gain.png")
    plt.close()

    loss_coef_by_voxel = regr_coef_by_voxel[..., 2]
    plot_loss = plot_volume(loss_coef_by_voxel, backdrop=np.nan)
    plt.imshow(plot_loss, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
    plt.colorbar()
    plt.title("Regression Coefficients by Voxel (Loss)")
    plt.savefig(path_result + "regr_coef_by_voxel_loss.png")
    plt.close()

    plot_dist2indiff = plot_volume(regr_coef_by_voxel[..., 3], backdrop=np.nan)
    plt.imshow(plot_dist2indiff, interpolation="nearest", cmap=nice_cmap,
               alpha=0.5)
    plt.colorbar()
    plt.title("Regression Coefficients by Voxel (Distance to Indifference)")
    plt.savefig(path_result + "regr_coef_by_voxel_dist2indiff.png")
    plt.close()


    # Compute the neural loss aversion
    neural_loss_aversion = -loss_coef_by_voxel - gain_coef_by_voxel
    
    # Inspect specifically the B ventral striatum
    mm_to_voxels = npl.inv(affine)
    BVS = nib.affines.apply_affine(mm_to_voxels, [3.6, 6.3, 3.9]).round()
    BVS = tuple(int(coordinate) for coordinate in BVS)
    neural_loss_aversion = neural_loss_aversion[BVS]
 
    # Save the results to the neural loss aversion repository file
    with open("results/neural_loss_aversion.txt", "a") as outfile:
        newline = run + "\t" + subject + "\t" + str(neural_loss_aversion) + "\n"
        outfile.write(newline)
