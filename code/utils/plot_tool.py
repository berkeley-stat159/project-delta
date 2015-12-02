"""
Plotting tools
--------------
"""
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
    
def show3Dimage(data, time=None):
    """
    Plot all horizontal slices of 4D fMRI image at a given point in time.

    Parameters:
    -----------
    data : np.ndarray
        4D array of fMRI data
    time : int
        The index (with respect to time) of the volume to plot

    Return:
    -------
    canvas: 2D array
        Canvas of horizontal slices of the brain at a given time
    """
    assert time == None or time <= data.shape[3]
    if time == None:
        length, width, depth = data.shape
    else:
        length, width, depth, timespan = data.shape
    len_side = int(np.ceil(np.sqrt(depth))) # Number slices per side of canvas
    canvas = np.zeros((length * len_side, width * len_side))
    depth_i = 0 # The ith slice with respect to depth
    for row in range(len_side):
        column = 0
        while column < len_side and depth_i < depth:
            if time == None:
                canvas[length * row:length * (row + 1), width * column:width * (column + 1)] = data[:, :, depth_i]
            else:
                canvas[length * row:length * (row + 1), width * column:width * (column + 1)] = data[:, :, depth_i, time]
            depth_i += 1
            column += 1
    return canvas

