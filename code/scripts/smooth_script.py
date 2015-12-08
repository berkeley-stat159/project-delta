"""" Script for smooth function 
Run with 
	python3 code/scripts/smooth_script.py

in the main project directory
""""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("code/utils")
from make_class import *

# Read in 4-dimensional filtered data
sub = run('001', '001', filtered_data = True)

# Use our .smooth method from class 
after_smooth = sub.smooth(sub.sigma)

#align the two plots
plt.subplot(211)

plt.subplot(212)
plot_bold_nii(data, slice_time)





