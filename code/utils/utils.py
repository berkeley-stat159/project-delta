import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

<<<<<<< HEAD
def plot_nii(filename):
    """Plot the middle slice of an anatomy image
    
    Parameters
    ----------
    filename: an nii file

    Output:
    -------
    A plot
    
    """
    img = nib.load(filename)
    data = img.get_data()
    if (len(data.shape)==3):
        plt.imshow(data[:,:,(data.shape[-1]//2)], interpolation="nearest")
    else:
        plt.imshow(data[:,:,(data.shape[-2]//2),(data.shape[-1]//2)], interpolation="nearest")

def outlier_prop(filename):
    img = nib.load(filename)
    data = img.get_data()
    data2d = np.reshape(data, (np.prod(data.shape[:-1]),data.shape[-1]))
    std = np.std(data2d, axis=0)
    IQR = np.percentile(std, 75) - np.percentile(std, 25)
    low_threshold = np.percentile(std, 25) - 1.5 * IQR
    high_threshold = np.percentile(std, 75) + 1.5 * IQR
    outlier_indices=np.nonzero((std < low_threshold) + (std > high_threshold))[0]
    return len(outlier_indices)/len(std)

def plot_bold_nii(filename, timepoint):
    """Plot all slices of a fMRI image in one plot at a specific time point

    Parameters:
    -----------
    filename: BOLD.nii.gz
    timepoint: the time point chose

    Return:
    -------
    None

    Note:
    -----
    The function produce a plot
    """
    img = nib.load(filename)
    data = img.get_data()
    assert timepoint <= data.shape[-1]
    plot_per_row = int(np.ceil(np.sqrt(data.shape[2])))
    frame = np.zeros((data.shape[0]*plot_per_row, data.shape[1]*plot_per_row))
    num_of_plots = 0
    for i in range(plot_per_row):
        j = 0
        while j < plot_per_row and num_of_plots < data.shape[2]:
            frame[data.shape[0]*i:data.shape[0]*(i+1), data.shape[1]*j:data.shape[1]*(j+1)] = data[:,:,num_of_plots,timepoint]
            num_of_plots+=1
            j+=1
    plt.imshow(frame, interpolation="nearest",cmap="gray")
    return None

=======
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

def plot_bold_nii(data, time):
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
        while plot < len_side and depth_i < depth:
            x_range = np.arange(length * row, width * (column + 1))
            y_start = np.arange(width * column, width * (column + 1))
            canvas[x_range, y_range] = data[..., depth_i, time]
            depth_i += 1
            column += 1
    plt.imshow(canvas, interpolation="nearest", cmap="gray")
    return None
>>>>>>> 389e0bd3ee306811cfd433fb19c1bf776c7ddff4
