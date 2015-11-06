import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def plot_nii(filename):
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