"""
This script contains tools that will be used to plot findings from statistical
analyses. Future Python scripts can take advantage of these utilities by
including the command
    sys.path.append("code/utils")
    from plot_tool import *
"""
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
    
def plot_volume(data, volume=None):
    """
    Plots all horizontal slices of a fMRI volume.

    Parameters
    ----------
    data : np.ndarray
        3- or 4-D array containing data imported from a .nii file
    volume : int, optional
        The index (with respect to time) of the volume of interest

    Return
    -------
    canvas : 2-D array
        Canvas depicting BOLD signal intensities of a given brain volume,
        organized left-to-right and top-to-bottom respectively in grid format
    """
    # Check assertions
    assert type(data) == np.ndarray, "data must be of type np.ndarray"
    if data.ndim == 4:
        assert volume != None and volume <= data.shape[3], "volume out of range"
        data = data[..., volume]
    elif data.ndim != 3:
        raise AssertionError("incorrect number of dimensions")
    # Extract data to be used for plotting
    length, width, depth = data.shape
    # Canvas is a grid: compute the number of slices to plot per side
    side_length = int(np.ceil(np.sqrt(depth)))
    canvas = np.zeros((length * side_length, width * side_length))
    # Plot slices iteratively: depth_i is the ith slice with respect to depth
    depth_i = 0
    for row in range(side_length):
        column = 0
        while column < side_length and depth_i < depth:
            canvas[length * row:length * (row + 1),
                   width * column:width * (column + 1)] = data[..., depth_i]
            column += 1
            depth_i += 1
    return canvas
