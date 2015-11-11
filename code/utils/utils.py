import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def plot_nii(filename):
    """
    Plots the middle slice of an anatomy image
    
    Parameters
    ----------
    filename: str
        The path to the file comtaining the anatomy 

    Returns
    -------
    Plot of the middle 
    
    """
    img = nib.load(filename)
    data = img.get_data()
    if (data.ndim == 3):
        
        plt.imshow(data[:,:,(data.shape[3]//2)], interpolation="nearest")
    else:
        mid_time = 
        plt.imshow(data[:,:,(data.shape[3]//2),(data.shape[4]//2)], interpolation="nearest")

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

