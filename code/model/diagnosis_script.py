""" Script to run diagnostic analysis on FMRI run"""
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.linalg as npl
from diagnostics import *

"""
* Load the image as an image object
* Load the image data from the image
* Drop the first four volumes, as we know these are outliers
"""

img = nib.load(".././bold.nii.gz")
data = img.get_data()
data = data[...,4:]

"""
Use your vol_std function to get the volume standard deviation values for the
remaining 169 volumes.
"""
vol_std_values = vol_std(data)
np.savetxt('vol_std_values.txt', vol_std_values)

"""
Use the iqr_outlier detection routine to get indices of outlier volumes.
"""
outlier_indices, lo_hi_thresh = iqr_outliers(vol_std_values)
np.savetxt('vol_std_outliers.txt', outlier_indices)

"""
Plot following
* The volume standard deviation values;
* The outlier points from the std values, marked on the plot with an 'o'
  marker;
* A horizontal dashed line at the lower IRQ threshold;
* A horizontal dashed line at the higher IRQ threshold;
"""
plt.plot(vol_std_values, c="b")
outlier = plt.scatter(outlier_indices, vol_std_values[outlier_indices],c="r")
lower_thres = plt.axhline(lo_hi_thresh[0], color="c",ls="--")
higher_thres = plt.axhline(lo_hi_thresh[1], color="g",ls="--")
plt.title("Volume Standard Deviation")
plt.xlabel('Volume Index')
plt.ylabel('Standard Deviation')
plt.xlim(0, 169)
plt.ylim(np.floor(min(vol_std_values)), np.ceil(max(vol_std_values)))
plt.legend((outlier, lower_thres, higher_thres),
			('Outliers', 'Lower IRQ threshold', 'Higher IRQ threshold'),
			loc=0)
plt.savefig('vol_std.png')
plt.close()

"""On the same plot, plot the following:
* The RMS vector;
* The identified outlier points marked with an `o` marker;
* A horizontal dashed line at the lower IRQ threshold;
* A horizontal dashed line at the higher IRQ threshold;"""

rmsd = vol_rms_diff(data)
rmsd_outlier_id, rmsd_thresh = iqr_outliers(rmsd)
plt.plot(rmsd, c="b")
rmsd_outlier = plt.scatter(rmsd_outlier_id, rmsd[rmsd_outlier_id],c="r")
lower_rmsd_thres = plt.axhline(rmsd_thresh[0], color="c",ls="--")
higher_rmsd_thres = plt.axhline(rmsd_thresh[1], color="g",ls="--")
plt.title("RMS Difference")
plt.xlabel('Difference Index')
plt.ylabel('rmsd')
plt.xlim(0, 169)
plt.legend((rmsd_outlier, lower_rmsd_thres, higher_rmsd_thres),
			('Outliers', 'Lower IRQ threshold', 'Higher IRQ threshold'),
			loc=0)
plt.savefig('vol_rms_outliers.png')
plt.close()


"""On the same plot, plot the following:
* The RMS vector with a 0 appended to make it have length the same as the
  number of volumes in the image data array;
* The identified outliers shown with an `o` marker;
* A horizontal dashed line at the lower IRQ threshold;
* A horizontal dashed line at the higher IRQ threshold;

IMPORTANT - save this plot as ``extended_vol_rms_outliers.png``
"""
edo_index = extend_diff_outliers(rmsd_outlier_id)
extend_rmsd = np.append(rmsd, 0)

plt.plot(extend_rmsd, c="b")
extend_rmsd_outlier = plt.scatter(edo_index, extend_rmsd[edo_index],c="r")
extend_lower_rmsd_thres = plt.axhline(rmsd_thresh[0], color="c",ls="--")
extend_higher_rmsd_thres = plt.axhline(rmsd_thresh[1], color="g",ls="--")
plt.title("Entended RMS Difference")
plt.xlabel('Difference Index')
plt.ylabel('rmsd')
plt.xlim(0, 169)
plt.legend((extend_rmsd_outlier, extend_lower_rmsd_thres, extend_higher_rmsd_thres),
			('Extended Outliers', 'Lower IRQ threshold', 'Higher IRQ threshold'),
			loc=0)
plt.savefig('extended_vol_rms_outliers.png')
plt.close()

""" Write the extended outlier indices to a text file."""
np.savetxt('extended_vol_rms_outliers.txt', edo_index)


convolved_1 = np.loadtxt('conv001.txt') #create col 1 to col 4
convolved_2 = np.loadtxt('conv002.txt')
convolved_3 = np.loadtxt('conv003.txt')
convolved_4 = np.loadtxt('conv004.txt')

convolved1 = convolved_1[4:] #covolve col 1 to col 4
convolved2 = convolved_2[4:]
convolved3 = convolved_3[4:]
convolved4 = convolved_4[4:]

N = len(convolved1)
X = np.ones((N, 5)) #make a metrix

X[:, 0] = convolved1 # put col 1 to col 4 to metrix
X[:, 1] = convolved2
X[:, 2] = convolved3
X[:, 3] = convolved4

data2d = np.reshape(data, (np.prod(data.shape[:-1]), -1))
data2d_trans = data2d.T
Xp = npl.pinv(X)
beta_hat = Xp.dot(data2d_trans)

MRSS_before = np.ones(data2d.shape[0])
res = data2d_trans - X.dot(beta_hat)
RSS = np.sum(res**2, axis=0)
df = X.shape[0] - npl.matrix_rank(X)
MRSS_before = RSS / df
print(np.mean(MRSS_before))



X_fixed = np.delete(X, edo_index, 0)
data2d_fixed = np.delete(data2d, edo_index, 1)
data2d_fixed_trans = data2d_fixed.T
Xp_fixed = npl.pinv(X_fixed)
beta_hat_fixed = Xp_fixed.dot(data2d_fixed_trans)

MRSS_after = np.ones(data2d_fixed.shape[0])
res_fixed = data2d_fixed_trans - X_fixed.dot(beta_hat_fixed)
RSS_fixed = np.sum(res_fixed**2, axis=0)
df_fixed = X_fixed.shape[0] - npl.matrix_rank(X_fixed)
MRSS_after = RSS_fixed / df_fixed
print(np.mean(MRSS_after))

mean_mrss = np.array([np.mean(MRSS_before), np.mean(MRSS_after)])
np.savetxt('mean_mrss_vals.txt', mean_mrss)

# Some final checks that you wrote the files with their correct names
from os.path import exists
assert exists('vol_std_values.txt')
assert exists('vol_std_outliers.txt')
assert exists('vol_std.png')
assert exists('vol_rms_outliers.png')
assert exists('extended_vol_rms_outliers.png')
assert exists('extended_vol_rms_outliers.txt')
assert exists('mean_mrss_vals.txt')
