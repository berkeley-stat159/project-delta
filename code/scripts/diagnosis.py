"""
Purpose
-------
This script runs a diagnostic analysis in bulk on the raw BOLD data of all runs
contained in the ds005 dataset. It preforms the task of finding outlier volumes,
with respect to the standard deviation among voxels in individual volumes.

It should export a total of six files:
- `vol_std_values.txt`: 
- `vol_std_outliers.txt`: 
- `vol_std.png`: 
- `vol_rms_outliers.png`: 
- `extd_vol_rms_outliers.png`: 
- `extd_vol_rms_outliers.txt`: 
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.linalg as npl
import sys

sys.path.append("code/utils")
from diagnostics import *
from hypothesis import *
from make_class import *
from plot_tool import *


# Begin by creating a collection of all subject IDs and all run IDs
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
IDs = list(zip([subject_ID for _ in range(3) for subject_ID in subject_IDs],
              [run_ID for _ in range(16) for run_ID in run_IDs]))


# We perform the outlined procedure for each run of each subject:
for ID in IDs:
    subject, run = ID


    # Create a directory to which the results will be saved
    path_result = "results/sub%s_run%s/diagnosis/" % (subject, run)
    try:
        os.makedirs(path_result)
    except OSError:
        if not os.path.isdir(path_result):
            raise


    # Extract all relevant data stored within the ds005 files:
    data = ds005(subject, run).filtered.data


    # Compute the standard deviation over all voxels for each volume
    vol_std_values = vol_std(data)
    np.savetxt(path_result + "vol_std_values.txt", vol_std_values)

    
    # Extract the indices of outlier volumes
    outlier_idx, lo_hi_thresh = iqr_outliers(vol_std_values)
    np.savetxt(path_result + "vol_std_outliers.txt", outlier_idx)


    # We plot the volume standard deviation values, marking each outlier point
    # with an 'o' and the two thresholds with horizontal dashed lines
    plt.plot(vol_std_values, c="b")
    outliers = plt.scatter(outlier_idx, vol_std_values[outlier_idx], c="r")
    lo_thresh = plt.axhline(lo_hi_thresh[0], color="c", ls="--")
    hi_thresh = plt.axhline(lo_hi_thresh[1], color="g", ls="--")
    plt.title("Volume Standard Deviation")
    plt.xlabel("Volume Index")
    plt.ylabel("Standard Deviation")
    plt.xlim(0, 240)
    plt.ylim(np.floor(min(vol_std_values)), np.ceil(max(vol_std_values)))
    plt.legend((outliers, lo_thresh, hi_thresh),
               ("Outliers", "Low IRQ Threshold", "High IRQ Threshold"), loc=0)
    plt.savefig(path_result + "vol_std.png")
    plt.close()


    # We make a new plot of the root-mean-square values, once again marking
    # each outlier with an 'o' and the thresholds with horizontal dashed lines
    rmsd = vol_rms_diff(data)
    rmsd_outlier_idx, rmsd_thresh = iqr_outliers(rmsd)
    plt.plot(rmsd, c="b")
    rmsd_outliers = plt.scatter(rmsd_outlier_idx, rmsd[rmsd_outlier_idx], c="r")
    lo_rmsd_thresh = plt.axhline(rmsd_thresh[0], color="c", ls="--")
    hi_rmsd_thresh = plt.axhline(rmsd_thresh[1], color="g", ls="--")
    plt.title("RMS Differences")
    plt.xlabel("Difference Index")
    plt.ylabel("RMS Difference")
    plt.xlim(0, 240)
    plt.legend((rmsd_outliers, lo_rmsd_thresh, hi_rmsd_thresh),
               ("Outliers", "Low IRQ threshold", "High IRQ threshold"), loc=0)
    plt.savefig(path_result + "vol_rms_outliers.png")
    plt.close()


    # We make one last plot of the extended difference outliers, once again with
    # each outlier marked with an 'o' and horizontal dashed lines at the
    # thresholds. Notice that we must append a 0 to the root-mean-square
    # differences so that its length will be equal to the number of volumes.
    edo_idx = extend_diff_outliers(rmsd_outlier_idx)
    extd_rmsd = np.append(rmsd, 0)
    plt.plot(extd_rmsd, c="b")
    extd_rmsd_outlier = plt.scatter(edo_idx, extd_rmsd[edo_idx], c="r")
    extd_lo_rmsd_thresh = plt.axhline(rmsd_thresh[0], color="c", ls="--")
    extd_hi_rmsd_thresh = plt.axhline(rmsd_thresh[1], color="g", ls="--")
    plt.title("Entended RMS Difference")
    plt.xlabel("Difference Index")
    plt.ylabel("RMS Difference")
    plt.xlim(0, 240)
    plt.legend((extd_rmsd_outlier, extd_lo_rmsd_thresh, extd_hi_rmsd_thresh),
               ("Extended Outliers", "Low IRQ Threshold", "High IRQ Threshold"),
               loc=0)
    plt.savefig(path_result + "extended_vol_rms_outliers.png")
    plt.close()


    # Lastly, in the spirit of good bookkeeping, we also save the extended
    # outlier indices to a plaintext file.
    np.savetxt(path_result + "extended_vol_rms_outliers.txt", edo_idx)

