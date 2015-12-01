"""
Plotting tools
--------------
1) show_slice: 2D plot of horizontal slice at specified depth (and specified time, if necessary)
2)
"""
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def show_slice(data, depth=None, time=None, color_option="gray"):
    """
    Plots the specified slice of an anatomy image
    (Show middle slice without specification)
    
    Parameters
    ----------
    data : np.ndarray
        fMRI data
    depth: int
        the postion for slicing
    time: int
        the time point for slicing
    color_option: str
        specifies color map, i.e. "gray", "bwr", "seismic", ...

    Returns
    -------
    2D plot of horizontal slice at specified depth (and specified time, if necessary).
    """
    if depth==None:
        depth = data.shape[2] // 2
    assert depth >= 0 and depth <= data.shape[2]
    if data.ndim==3:
        fig = plt.imshow(data[..., depth], interpolation="nearest", cmap=color_option)
        plt.colorbar()
    else:
        if time==None:
            time = data.shape[3] // 2
        assert time >= 0 and time <= data.shape[3]
        plt.imshow(data[..., depth, time], interpolation="nearest", cmap=color_option)
        plt.colorbar()

def show3Dimage(data, time=None, color_option="gray"):
    """
    Plot all horizontal slices of 4D fMRI image at a given point in time.

    Parameters:
    -----------
    data : np.ndarray
        4D array of fMRI data
    time : int
        The index (with respect to time) of the volume to plot
    color_option: str
        specifies color map, i.e. "gray", "bwr", "seismic", ...

    Return:
    -------
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
    plt.imshow(canvas, interpolation="nearest", cmap=color_option)
    plt.colorbar()

