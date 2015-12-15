""" Script to generate neuroimages on FMRI run using nilearn package

Requirements:
------------
Please install nilearn first by running "pip install -U --user nilearn"

Goals:
-----
	To visualize the activated regions on the brain template for one subject

Including:
---------
1) Create brain template
2) Simply plot all t scores and p values
3) Locate activated regions in one subject
4) Glass Map of p values
"""
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.linalg as npl
from matplotlib import colors
from nilearn import image
from nilearn.plotting import plot_stat_map,plot_glass_brain,plot_anat
from nilearn.input_data import NiftiMasker
from nilearn.image.resampling import coord_transform

import sys
sys.path.append("code/utils")
from make_class import *
from plot_tool import *

sub = ds005("001","001")
data = sub.filtered.data
affine = sub.filtered.affine
img = sub.filtered.img
smooth_img = image.smooth_img(img, 6) # smoothed image
smooth_data = smooth_img.get_data()
smooth_data = smooth_data[...,4:]
vol_shape, n_trs = smooth_data.shape[:-1], smooth_data.shape[-1]
file_path = "results/run001/glm/sub001/"

"""
1) Create brain template using mean_img
"""
mean_smooth_img = image.mean_img(smooth_img, affine, smooth_img.shape[:-1])
plot_anat(mean_smooth_img, cut_coords=(0,0,0), title='Smoothed mean brain image')
plt.savefig(file_path+'brain_template.png')
plt.close()

"""
2) Simply plot all t scores and p values
"""
nice_cmap_values = np.loadtxt("code/scripts/actc.txt")
nice_cmap = colors.ListedColormap(nice_cmap_values, "actc")

t_map_gain_filename = file_path+"t_stat_gain.nii.gz"
t_map_loss_filename = file_path+"t_stat_loss.nii.gz"
t_map_dist_filename = file_path+"t_stat_dist2indiff.nii.gz"

p_val_gain_filename = file_path+"p_value_gain.nii.gz"
p_val_loss_filename = file_path+"p_value_loss.nii.gz"
p_val_dist_filename = file_path+"p_value_dist2indiff.nii.gz"

plot_gain = plot_volume(nib.load(t_map_gain_filename).get_data(),backdrop=np.nan)
plt.imshow(plot_gain, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("T Scores (Gain)")
plt.savefig(file_path + "t_map_gain_slice.png")
plt.close()

plot_loss = plot_volume(nib.load(t_map_loss_filename).get_data())
plt.imshow(plot_loss, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("T Scores (Loss)")
plt.savefig(file_path + "t_map_loss_slice.png")
plt.close()

plot_dist = plot_volume(nib.load(t_map_dist_filename).get_data())
plt.imshow(plot_dist, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("T Scores (Dist)")
plt.savefig(file_path + "t_map_dist_slice.png")
plt.close()

plot_gain_p = plot_volume(nib.load(p_val_gain_filename).get_data())
plt.imshow(plot_gain_p, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("P values (Gain)")
plt.savefig(file_path + "p_val_gain_slice.png")
plt.close()

plot_loss_p = plot_volume(nib.load(p_val_loss_filename).get_data())
plt.imshow(plot_loss_p, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("P values (Loss)")
plt.savefig(file_path + "p_val_loss_slice.png")
plt.close()

plot_dist_p = plot_volume(nib.load(p_val_dist_filename).get_data())
plt.imshow(plot_dist_p, interpolation="nearest", cmap=nice_cmap, alpha=0.5)
plt.colorbar()
plt.title("P values (Dist)")
plt.savefig(file_path + "p_val_dist_slice.png")
plt.close()

"""
3) Locate activated regions in one subject
	a) t map
	b) p-value
"""
cut_coords = (0, 0, 0)
# t-score
plot_stat_map(t_map_gain_filename, bg_img=mean_smooth_img,
                       threshold="auto", title="t_score_map: gain",
                       cut_coords=cut_coords, output_file=file_path+"t_scores_gain.png", black_bg=False)

plot_stat_map(t_map_loss_filename, bg_img=mean_smooth_img,
                       threshold="auto", title="t_score_map: loss",
                       cut_coords=cut_coords, output_file=file_path+"t_scores_loss.png", black_bg=False)

plot_stat_map(t_map_dist_filename, bg_img=mean_smooth_img,
                       threshold="auto", title="t_score_map: dist",
                       cut_coords=cut_coords, output_file=file_path+"t_scores_dist.png",black_bg=False)


# p value
# gain
p_values1 = nib.load(p_val_gain_filename).get_data()
log_p_values1 = -np.log10(p_values1)
log_p_values1[np.isnan(log_p_values1)] = 0.
log_p_values1[log_p_values1 > 10] = 10.
log_p_values1[log_p_values1 < 1.3] = 0.
plot_stat_map(nib.Nifti1Image(log_p_values1,affine),mean_smooth_img, title="p-values: gain", black_bg=False, 
  display_mode="z", cut_coords= 6,output_file=file_path+"p_val_gain_brain.png")

# loss
p_values2 = nib.load(p_val_loss_filename).get_data()
log_p_values2 = -np.log10(p_values2)
log_p_values2[np.isnan(log_p_values2)] = 0.
log_p_values2[log_p_values2 > 10] = 10.
log_p_values2[log_p_values2 < 1.3] = 0.
plot_stat_map(nib.Nifti1Image(log_p_values2,affine),mean_smooth_img, title="p-values: loss", black_bg=False, 
  display_mode="z", cut_coords= 6,output_file=file_path+"p_val_loss_brain.png")

# distance
p_values3 = nib.load(p_val_dist_filename).get_data()
log_p_values3 = -np.log10(p_values3)
log_p_values3[np.isnan(log_p_values3)] = 0.
log_p_values3[log_p_values3 > 10] = 10.
log_p_values3[log_p_values3 < 1.3] = 0.
plot_stat_map(nib.Nifti1Image(log_p_values3,affine),mean_smooth_img, title="p-values: dist", black_bg=False, 
  display_mode="z", cut_coords= 6,output_file=file_path+"p_val_dist_brain.png")

"""
4) Glass Map of p values
Glass brain visualization. 
By default plots maximum intensity projection of the absolute values. 
"""
thres = - np.log10(0.05)

nifti_masker1 = NiftiMasker(smoothing_fwhm=5,memory='nilearn_cache', memory_level=1)
p_value_gain = nifti_masker1.fit_transform(p_val_gain_filename)
neg_log_p_value_gain = - np.log10(p_value_gain)
neg_log_p_value_gain[np.isnan(neg_log_p_value_gain)] = 0.
neg_log_p_value_gain[neg_log_p_value_gain > 10] = 10.
neg_log_p_value_gain_unmasked = nifti_masker1.inverse_transform(neg_log_p_value_gain)

nifti_masker2 = NiftiMasker(smoothing_fwhm=5,memory='nilearn_cache', memory_level=1)
p_value_loss = nifti_masker2.fit_transform(p_val_loss_filename)
neg_log_p_value_loss = - np.log10(p_value_loss)
neg_log_p_value_loss[np.isnan(neg_log_p_value_loss)] = 0.
neg_log_p_value_loss[neg_log_p_value_loss > 10] = 10.
neg_log_p_value_loss_unmasked = nifti_masker2.inverse_transform(neg_log_p_value_loss)

nifti_masker3 = NiftiMasker(smoothing_fwhm=5,memory='nilearn_cache', memory_level=1)
p_value_dist = nifti_masker3.fit_transform(p_val_dist_filename)
neg_log_p_value_dist = - np.log10(p_value_dist)
neg_log_p_value_dist[np.isnan(neg_log_p_value_dist)] = 0.
neg_log_p_value_dist[neg_log_p_value_dist > 10] = 10.
neg_log_p_value_dist_unmasked = nifti_masker3.inverse_transform(neg_log_p_value_dist)

plot_glass_brain(neg_log_p_value_gain_unmasked, output_file=file_path+"glass_brain_gain.png",colorbar=True)
plot_glass_brain(neg_log_p_value_loss_unmasked, output_file=file_path+"glass_brain_loss.png",colorbar=True)
plot_glass_brain(neg_log_p_value_dist_unmasked, output_file=file_path+"glass_brain_dist.png",colorbar=True)

plot_glass_brain(t_map_gain_filename, output_file=file_path+"t_glass_brain_gain.png",colorbar=True)
plot_glass_brain(t_map_loss_filename, output_file=file_path+"t_glass_brain_loss.png",colorbar=True)
plot_glass_brain(t_map_dist_filename, output_file=file_path+"t_glass_brain_dist.png",colorbar=True)
