from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def load_data(filename):
    """
    Loads the data in a file, given its path.
    
    Parameters
    ----------
    filename : str
        The path to the file containing fMRI data
    
    Returns
    -------
    np.ndarray of data to be saved to the python environment
    """
    return nib.load(filename).get_data()

def plot_nii(data):
    """
    Plots the middle slice of an anatomy image
    
    Parameters
    ----------
    data : np.ndarray
        fMRI data

    Returns
    -------
    2D plot of horizontal slice at center depth (and center time, if necessary).
    """
    mid_depth = data.shape[2] // 2
    if (data.ndim==3):
        plt.imshow(data[..., mid_depth], interpolation="nearest")
    else:
        mid_time = data.shape[3] // 2
        plt.imshow(data[..., mid_depth, mid_time], interpolation="nearest")

def outlier_prop(data, iqs_scale=1.5):
    """
    Computes the proportion of outliers with respect to time in a data set.
    
    Parameters
    ----------
    data : np.ndarray
        fMRI data
    iqr_scale : float, optional
        Multiples of IQR outside which to consider a point an outlier
    
    Returns
    -------
    Proportion of volumes considered outliers in time
    """
    data2d = data.reshape([-1, data.shape[3]])
    std = data2d.std(axis=0)
    IQR = np.percentile(std, 75) - np.percentile(std, 25)
    low_threshold = np.percentile(std, 25) - iqr_scale * IQR
    high_threshold = np.percentile(std, 75) + iqr_scale * IQR
    outlier_i = np.nonzero((std < low_threshold) + (std > high_threshold))[0]
    return len(outlier_i) / len(std)

def plot_bold_nii(data, time, color=False):
    """
    Plot all horizontal slices of fMRI image at a given point in time.

    Parameters:
    -----------
    data : np.ndarray
        4D array of fMRI data
    time : int
        The index (with respect to time) of the volume to plot

    Return:
    -------
    Canvas of horizontal slices of the brain at a given time
    """
    assert time <= data.shape[3]
    length, width, depth, timespan = data.shape
    len_side = int(np.ceil(np.sqrt(depth))) # Number slices per side of canvas
    canvas = np.zeros((length * len_side, width * len_side))
    depth_i = 0 # The ith slice with respect to depth
    for row in range(len_side):
        column = 0
        while column < len_side and depth_i < depth:
            canvas[length * row:length * (row + 1), width * column:width * (column + 1)] = data[..., depth_i, time]
            depth_i += 1
            column += 1
    if color:
        plt.imshow(canvas, interpolation="nearest")
    else:
        plt.imshow(canvas, interpolation="nearest", cmap="gray")
    return None
