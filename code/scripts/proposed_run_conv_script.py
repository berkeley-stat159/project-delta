"""
Purpose:
-------
This script will produce convolved hemodynamic predictions for one run per
subject. Each of four conditions selected from the behavioral data (gain, loss,
response, response time) will be used separately and the results will
be saved to a new txt file. Lastly, this script will generate four BOLD signals
(hemodynamic prediction) over time plots for each of the four conditions.

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
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.append("code/utils")
from make_class import run

sys.path.append("code/model")
from convolution import *

TR = 2
num_TRs = 240
TR_divs = 100

# 1) Read in BOLD and behavioral data
sub = run("001", "001")

# 2) Get a neural time course for each condition
tc_gain = sub.time_course("gain", step_size=1/TR_divs)
tc_loss = sub.time_course("loss", step_size=1/TR_divs)
tc_resp = sub.time_course("resp", step_size=1/TR_divs)
tc_resp_time = sub.time_course("resp_time", step_size=1/TR_divs)

# 3) Convolve with hrf
hrf_times = np.arange(0, 30, TR / tr_divs)
hrf_at_hr = hrf(hrf_times)

pred_gain = np.convolve(tc_gain, hrf_at_hr)[:len(tc_gain)]
pred_loss = np.convolve(tc_loss, hrf_at_hr)[:len(tc_loss)]
pred_conf = np.convolve(tc_resp, hrf_at_hr)[:len(tc_conf)]
pred_resp_time = np.convolve(tc_resp_time, hrf_at_hr)[:len(tc_resp_time)]

# 4) Plots
all_TR_times = np.arange(0, num_TRs, 1/TR_divs) * TR

plt.subplot(221)
plt.plot(all_TR_times, pred_gain)
plt.title("Condition 1: Gain")

plt.subplot(222)
plt.plot(all_TR_times, pred_loss)
plt.title("Condition 2: Loss")

plt.subplot(223)
plt.plot(all_TR_times, pred_resp)
plt.title("Condition 3: Response")

plt.subplot(224)
plt.plot(all_TR_times, pred_resp_time)
plt.title("Condition 4: Response Time")

plt.savefig('convolution4cond_v1.png')
plt.close()

# 5) Save as txt files
index = np.arange(0, num_TRs * TR_divs, TR_divs)
np.savetxt("conv001.txt", pred_gain[index])
np.savetxt("conv002.txt", pred_loss[index])
np.savetxt("conv003.txt", pred_conf[index])
np.savetxt("conv004.txt", pred_resp_time[index])

