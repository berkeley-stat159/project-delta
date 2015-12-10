"""
This script contains code that will assist in calculating the correlation
coefficient for individual voxels across the dimension of time. Future Python
scripts can take advantage of the correlation module by including the command
    sys.path.append("code/utils")
    from correlation import *
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os, sys
from scipy.ndimage.filters import gaussian_filter


def correlation(obj, img, regressor):
    """
    Calculates the correlation coefficient of the BOLD signal with a single
    regressor for each voxel across time.
    
    Parameters
    ----------
    ds005_obj : object
        Instance of class ds005
    img_type : str
        Dataset of interest: select from "raw" and "filtered"
    regressor : str
        Name of regressor whose correlation with the BOLD data is of interest:
        select from "gain", "loss", "dist2indiff"
        
    Return
    ------
    corr : np.ndarray
        Array of shape (run.data.shape[:3],), where each value in 3-D space
        is the corresponding voxel's correlation coefficient of the BOLD
        signal with the specified regressor over time
    """
    type_obj = str(type(obj))
    assert type_obj == "<class 'make_class.ds005'>", "not an instance of ds005"
    assert img in ["raw", "filtered"], "invalid input to argument img"
    data = obj.raw.data if img == "raw" else obj.filtered.data
    time_course = obj.time_course(regressor)
    n_voxels, n_volumes = np.prod(data.shape[:3]), data.shape[3]
    voxels = data.reshape(n_voxels, n_volumes)
    corr_1d = [np.corrcoef(voxel, time_course)[0, 1] for voxel in voxels]
    corr = np.reshape(corr_1d, data.shape[:3])
    return corr
