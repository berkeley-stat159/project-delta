import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys
from smoothing import *

#Change path to utils.py to look for modules
#sys.path.append("../utils")
from utils import *
data = load_data("bold.nii")

#select a random time slice 
slice_time = 3
sigma = 1
after_smooth = smooth(data, sigma, slice_time)

#align the two plots
plt.subplot(211)
plot_bold_nii(after_smooth, slice_time)
plt.subplot(212)
plot_bold_nii(data, slice_time)





