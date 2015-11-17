"""
Utility functions

"""
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

def read_txt_files(filename):
    """read from behav.txt and covert data to array format

    Parameter:
    ---------
    filename: behavdata.txt file

    Return:
    ------
    val_arr: 
        values read from behavdata.txt and sorted as an array
    """
    with open(filename,"r") as infile:
        lines = infile.readlines()
    lines = lines[1:]
    val = [line.split() for line in lines]
    val_arr = np.array(val, dtype=float)
    return val_arr

def construct_mat(array, includeED=True):
    """Construct the design matrix using the array return from the
    read_files function.

    Parameter:
    ----------
    array: 2-D array
        return by read_files
    includeED: boolean (True or False)
        True (default): include Euclidean distance as the 3rd regressor (4th column of design_matrix)
        False: not include. Only two regressors (gain and loss) are included.

    Return:
    ------
    design_matrix: 2-D array 
        The first column is always intercept, and the last column is the response (0 and 1)
        The column names are (intercept, gain, loss, distance(if includeED=True), response) 
    """
    array = array[array[:,-2]!=-1,:]
    if includeED:
        mat = array[:,[1,2,4,5]]
        diagOfGambleMat = np.array([np.sum(mat[:,2]==1), np.sum(mat[:,2]==3)])
        distance = np.sqrt(np.sum((mat[:,:2]-diagOfGambleMat)**2, axis=1))
        design_matrix = np.ones((mat.shape[0],mat.shape[1]+1))
        design_matrix[:,1:] = mat
        design_matrix[:,3] = distance
    else:
        mat = array[:,[1,2,5]]
        design_matrix = np.ones((mat.shape[0],mat.shape[1]+1))
        design_matrix[:,1:] = mat
    return design_matrix
