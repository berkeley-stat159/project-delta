"""
This script contains tools for performing diagnostics analyses on fMRI data.
Future Python scripts can take advantage of these functions by including the
command:
    sys.path.append("code/utils")
    from diagnostics import *
"""
from __future__ import division, print_function, absolute_import
import numpy as np


def vol_std(data):
    """
    Computes the standard deviation over all voxels for each volume.

    Parameters
    ----------
    data : np.ndarray
        4-D array from fMRI run with last axis indexing volumes. Call the shape
        of this array (M, N, P, T) where T is the number of volumes.

    Return
    ------
    std_values : np.ndarray
        1-D array of shape (T,) where the ith value gives the standard deviation
        over all voxels contained in the ith volume.
    """
    data2d = np.reshape(data, (np.prod(data.shape[:3]), data.shape[3]))
    return np.std(data2d, axis=0)

def iqr_outliers(arr_1d, iqr_scale=1.5):
    """
    Return the indices of outliers identified by interquartile range.

    Parameters
    ----------
    arr_1d : np.ndarray
        1-D array from which we will identify outliers values.
    iqr_scale : float, optional
        Coefficient used to determine the weight of the IQR in determining the
        high and low thresholds.

    Return
    ------
    outlier_indices : np.ndarray
        Array containing indices of volumes considered to be outliers.
    lo_hi_thresh : tuple
        Tuple of format (low threshold, high threshold), where
        - Low threshold = first quartile - iqr_scale * IQR
        - High threshold = third quartile + iqr_scale * IQR
    """
    IQR = np.percentile(arr_1d, 75) - np.percentile(arr_1d, 25)
    low_threshold = np.percentile(arr_1d, 25) - iqr_scale * IQR
    high_threshold = np.percentile(arr_1d, 75) + iqr_scale * IQR
    outlier_indices = np.nonzero((arr_1d < low_threshold) +
                                 (arr_1d > high_threshold))[0]
    return (outlier_indices, (low_threshold, high_threshold))

def vol_rms_diff(arr_4d):
    """
    Computes the root-mean-square of differences between sequential volumes.

    Parameters
    ----------
    data : np.ndarray
        4-D array of fMRI data with last axis indexing volumes. Call the shape
        of this array (M, N, P, T) where T is the number of volumes.

    Return
    ------
    rms_values : np.ndarray
        Array of shape (T - 1,) in which the ith element is the square root of
        the mean (across voxels) of the squared difference between the ith
        volume and the (i + 1)th volume
    """
    diff = np.subtract(arr_4d[..., 1:], arr_4d[..., :-1])
    diff2d = np.reshape(diff, (np.prod(diff.shape[:3]), diff.shape[3]))
    rms_values = np.sqrt(np.mean(diff2d ** 2, axis=0))
    return rms_values

def extend_diff_outliers(diff_indices):
    """
    Extend difference-based outlier indices `diff_indices` by pairing

    Parameters
    ----------
    diff_indices : np.ndarray
        1-D array of indices of differences detected to be outliers. A
        difference index of i refers to the difference between the ith volume
        and (i + 1)th volume.

    Return
    ------
    extended_indices : np.ndarray
        Array where each index j in diff_indices has been replaced by two
        indices, j and (j + 1), unless (j + 1) is already present in
        diff_indices. For example:

    >>> diff_indices = np.array([3, 7, 8, 12, 20])
    >>> extend_diff_outlier(diff_indices)
    np.array([3, 4, 7, 8, 9, 12, 13, 20, 21])
    """
    return np.unique(np.append(diff_indices, diff_indices + 1))
