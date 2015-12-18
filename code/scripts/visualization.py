""" Script to generate neuroimages on fMRI run using nilearn package

Requirements:
------------
Please install `nilearn` first by running "pip install -U --user nilearn"
in command line

Purpose:
--------
	To visualize the activated regions on the brain template 16 subjects

Including:
---------
1) Create brain template
2) Locate activated regions using t-statistic
3) Glass Map of t t-statistic

Outputs:
--------
Total seven plots are generated in each run:
- `brain_template.png`
- `t_scores_gain.png`
- `t_scores_loss.png`
- `t_scores_dist.png`
- `t_glass_brain_gain.png`
- `t_glass_brain_loss.png`
- `t_glass_brain_dist.png`

"""
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.linalg as npl
from matplotlib import colors
from nilearn import image
from nilearn.plotting import plot_stat_map,plot_glass_brain,plot_anat

import sys
sys.path.append("code/utils")
from make_class import *
from plot_tool import *

# Create a collection of all subject IDs and all run IDs
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
IDs = list(zip([run_ID for _ in range(16) for run_ID in run_IDs],
               [subject_ID for _ in range(3) for subject_ID in subject_IDs]))
IDs.sort()

for ID in IDs:
    run, subject = ID

    sub = ds005(subject,run)
    data = sub.filtered.data
    affine = sub.filtered.affine
    img = sub.filtered.img
    smooth_img = image.smooth_img(img, 6) # smoothed image

    path_result = "results/run%s/visualization/sub%s/" % ID
    file_path = "results/run%s/glm/sub%s/" % ID

    try:
        os.makedirs(path_result)
    except OSError:
        if not os.path.isdir(path_result):
            raise
    
    t_map_gain_filename = file_path+"t_stat_gain.nii.gz"
    t_map_loss_filename = file_path+"t_stat_loss.nii.gz"
    t_map_dist_filename = file_path+"t_stat_dist2indiff.nii.gz"

    """
    1) Create brain template using mean_img
    """
    mean_smooth_img = image.mean_img(smooth_img, affine, smooth_img.shape[:-1])
    plot_anat(mean_smooth_img, cut_coords=(0,0,0), title='Smoothed mean brain image')
    plt.savefig(path_result+'brain_template.png')
    plt.close()

    """
    2) Locate activated regions: t-statistic map
    """
    cut_coords = (0, 0, 0)
    # t-score
    plot_stat_map(t_map_gain_filename, bg_img=mean_smooth_img,
                           threshold="auto", title="t_score_map: gain",
                           cut_coords=cut_coords, output_file=path_result+"t_scores_gain.png", black_bg=False)

    plot_stat_map(t_map_loss_filename, bg_img=mean_smooth_img,
                           threshold="auto", title="t_score_map: loss",
                           cut_coords=cut_coords, output_file=path_result+"t_scores_loss.png", black_bg=False)

    plot_stat_map(t_map_dist_filename, bg_img=mean_smooth_img,
                           threshold="auto", title="t_score_map: dist",
                           cut_coords=cut_coords, output_file=path_result+"t_scores_dist.png",black_bg=False)

    """
    3) Glass Map of t values
    Glass brain visualization. 
    By default plots maximum intensity projection of the absolute values. 
    """
    plot_glass_brain(t_map_gain_filename, output_file=path_result+"t_glass_brain_gain.png",colorbar=True)
    plot_glass_brain(t_map_loss_filename, output_file=path_result+"t_glass_brain_loss.png",colorbar=True)
    plot_glass_brain(t_map_dist_filename, output_file=path_result+"t_glass_brain_dist.png",colorbar=True)
