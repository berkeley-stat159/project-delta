"""
Purpose:
-------
In this run_conv_script, we will produce convolved hemodynamic prediction into seperated txt 
files for all four conditions we select from hehavdata.txt: gain, loss, confidence, response time 
for one run for one subject.
This script will also generate four BOLD signals (hemodynamic prediction) over time plots for all
four conditions.

Steps:
-----
1) Read in behavdata.txt and get rid of invalid data (where response is -1)
2) Extract 4 conditions, onsets, duration from hehav
3) Get neural time course for each condition
4) Convolve with HRF
5) Plots
6) Save to txt files
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".././utils")
from stimuli import *
from utils_functions import *
sys.path.append(".././model")
from convolution import *

TR = 2
n_trs = 240
tr_divs = 100

# 1) Read in behavdata.txt and get rid of invalid data (where response is -1)
behav = read_txt_files(".././behavdata.txt")
behav = behav[behav[:,-2]!=-1,:]

# 2) Extract 4 conditions, onsets, duration from hehav
onsets = behav[:,0]
durations = np.ones(len(onsets))*3
gain = behav[:,1]
loss = behav[:,2]
conf = behav[:,4]
restime = behav[:,6]

# 3) Get neural time course for each condition
neural_gain = neural_highres(onsets,durations,gain)
neural_loss = neural_highres(onsets,durations,loss)
neural_conf = neural_highres(onsets,durations,conf)
neural_restime = neural_highres(onsets,durations,restime)

# 4) Convolve with hrf
hrf_times = np.arange(0,30,TR/tr_divs)
hrf_at_hr = hrf(hrf_times)

hemo_pred_gain = np.convolve(neural_gain, hrf_at_hr)[:len(neural_gain)]
hemo_pred_loss = np.convolve(neural_loss, hrf_at_hr)[:len(neural_loss)]
hemo_pred_conf = np.convolve(neural_conf, hrf_at_hr)[:len(neural_conf)]
hemo_pred_restime = np.convolve(neural_restime, hrf_at_hr)[:len(neural_restime)]

# 5) Plots
all_tr_times = np.arange(0, n_trs, 1/tr_divs)*TR

plt.subplot(221)
plt.plot(all_tr_times, hemo_pred_gain)
plt.title("Condition 1: Gain")

plt.subplot(222)
plt.plot(all_tr_times, hemo_pred_loss)
plt.title("Condition 2: Loss")

plt.subplot(223)
plt.plot(all_tr_times, hemo_pred_conf)
plt.title("Condition 3: Confidence")

plt.subplot(224)
plt.plot(all_tr_times, hemo_pred_restime)
plt.title("Condition 4: Response Time")

plt.savefig('convolution4cond_v1.png')
plt.close()

#6) save tp txt files
index = np.arange(0,n_trs*tr_divs, tr_divs)
np.savetxt("conv001.txt", hemo_pred_gain[index])
np.savetxt("conv002.txt", hemo_pred_loss[index])
np.savetxt("conv003.txt", hemo_pred_conf[index])
np.savetxt("conv004.txt", hemo_pred_restime[index])

