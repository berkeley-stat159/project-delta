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
2) Extract 3 conditions and Get neural time course for each condition
3) Convolve with HRF
4) Plots
5) Save to txt files
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("code/utils")
from make_class import *
from convolution import *

TR = 2
n_trs = 240
tr_divs = 100

# 1) Read in behavdata.txt and get rid of invalid data (where response is -1)
sub = run("001","001")

# 2) Extract 3 conditions: gain, loss, dist_from_indiff
neural_gain = sub.time_course("gain", step_size=0.02) # where 0.02 = TR/tr_divs
neural_loss = sub.time_course("loss", step_size=0.02)
neural_dist = sub.time_course("dist_from_indiff", step_size=0.02)

# 3) Convolve with hrf
hrf_times = np.arange(0,30,TR/tr_divs)
hrf_at_hr = hrf(hrf_times)

hemo_pred_gain = np.convolve(neural_gain, hrf_at_hr)[:len(neural_gain)]
hemo_pred_loss = np.convolve(neural_loss, hrf_at_hr)[:len(neural_loss)]
hemo_pred_dist = np.convolve(neural_dist, hrf_at_hr)[:len(neural_dist)]

# 4) Plots
all_tr_times = np.arange(0, n_trs, 1/tr_divs)*TR

plt.figure(figsize=(15, 10), dpi=100)
plt.subplot(231)
plt.plot(all_tr_times, hemo_pred_gain)
plt.xlabel("Time")
plt.ylabel("Hemodynamic Response")
plt.title("Condition 1: Gain")

plt.subplot(232)
plt.plot(all_tr_times, hemo_pred_loss)
plt.xlabel("Time")
plt.ylabel("Hemodynamic Response")
plt.title("Condition 2: Loss")

plt.subplot(233)
plt.plot(all_tr_times, hemo_pred_dist)
plt.xlabel("Time")
plt.ylabel("Hemodynamic Response")
plt.title("Condition 3: Distance from Indifference")

plt.subplot(234)
plt.plot(all_tr_times, neural_gain)
plt.xlabel("Time")
plt.ylabel("Neural Prediction")
plt.title("Condition 1: Gain")
plt.subplot(235)
plt.plot(all_tr_times, neural_loss)
plt.xlabel("Time")
plt.ylabel("Neural Prediction")
plt.title("Condition 2: Loss")
plt.subplot(236)
plt.plot(all_tr_times, neural_dist)
plt.xlabel("Time")
plt.ylabel("Neural Prediction")
plt.title("Condition 3: Distance from Indifference")

path = "results_sub1/"
if not os.path.exists(path):
	os.makedirs(path)

plt.savefig(path+'convolution3cond.png',dpi=500)
plt.close()

#5) save tp txt files
index = np.arange(0,n_trs*tr_divs, tr_divs)
np.savetxt(path+"conv001.txt", hemo_pred_gain[index])
np.savetxt(path+"conv002.txt", hemo_pred_loss[index])
np.savetxt(path+"conv003.txt", hemo_pred_dist[index])

