"""
In this run_conv_script, we will produce convolved BOLD signals into 
seperated txt files for all four condition files for one run for one subject. 
This script will also generate four BOLD signals over time plots for all 
four conditions
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from stimuli import events2neural
from convolution import convolve

TR = 2
n_vols = 240
duration = 3/TR

all_tr_times = np.arange(240)*2

neural1 = events2neural(".././cond001.txt", TR, n_vols)
neural2 = events2neural(".././cond002.txt", TR, n_vols)
neural3 = events2neural(".././cond003.txt", TR, n_vols)
neural4 = events2neural(".././cond004.txt", TR, n_vols)

convolved1 = convolve(neural1, TR, n_vols, duration)
np.savetxt("conv001.txt", convolved1)
convolved2 = convolve(neural2, TR, n_vols, duration)
np.savetxt("conv002.txt", convolved2)
convolved3 = convolve(neural3, TR, n_vols, duration)
np.savetxt("conv003.txt", convolved3)
convolved4 = convolve(neural4, TR, n_vols, duration)
np.savetxt("conv004.txt", convolved4)

plt.subplot(221)
plt.plot(all_tr_times, convolved1)
plt.plot(all_tr_times, neural1)
plt.title("Condition 1")

plt.subplot(222)
plt.plot(all_tr_times, convolved2)
plt.plot(all_tr_times, neural2)
plt.title("Condition 2")

plt.subplot(223)
plt.plot(all_tr_times, convolved3)
plt.plot(all_tr_times, neural3)
plt.title("Condition 3")

plt.subplot(224)
plt.plot(all_tr_times, convolved4)
plt.plot(all_tr_times, neural4)
plt.title("Condition 4")

plt.savefig('convolution4cond.png')
plt.close()

