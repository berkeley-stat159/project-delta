"""
This script contains code that performs a number of computations that tend to be
very useful in statistical analysis. Future Python scripts can take advantage of
this module by including the command
    sys.path.append("code/utils")
    from stat_utils import *
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.linalg as npl
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

def glm_util(design_matrix, response):
    """
    Fits a generalized linear model to a set of training data.

    Parameters
    ----------
    design_matrix : np.ndarray
        2-D array with rows that correspond to observations and columns that
        correspond to regressors. Let the shape of design_matrix be (N, P)
    response : np.ndarray
        1- or 2-D array representing the response variable. Let the shape of
        response be (N, X)

    Return
    ------
    regression_coefficients : np.ndarray
        Array of shape (P, X) containing the coefficient of the regressor for
        the individual data point
    df : int
        Degrees of freedom, which is the difference of the number of independent
        regressors from the number of observations
    MRSS : float
        Mean residual sum of squares, a commmonly used measure of a predictive
        model's accuracy (lower is better)
    """
    regression_coefficients = npl.pinv(design_matrix).dot(response)
    prediction = design_matrix.dot(regression_coefficients)
    error = response - prediction
    RSS = (error ** 2).sum(0)
    df = int(design_matrix.shape[0] - npl.matrix_rank(design_matrix))
    MRSS = RSS / df
    return (regression_coefficients, df, MRSS)
