"""
Purpose
-------
This script applies smoothing with a Gaussian kernel in bulk to both the raw and
the filtered BOLD signal data. The kernel used 

"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.append("code/utils")
from make_class import *
from plot_tool import *

image_location = 

# Read in 4-dimensional filtered data
sub = run('001', '001', filtered_data = True)

# Use our .smooth method from class 
after_smooth = sub.smooth(sub.sigma_filtered)

#align the two plots
plt.imshow(show3Dimage(after_smooth, 50))
plt.colorbar()
plt.title('After Smoothed Data')
plt.savefig(image_location+'Smoothed_data.png')
plt.close()

plt.imshow(show3Dimage(sub.data, 50))
plt.colorbar()
plt.title('Original Data')
plt.savefig(image_location+'Original_data.png')
plt.close()