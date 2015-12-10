"""
Purpose:
-------
This script allows production, for each run of each subject, of convolved
hemodynamic response predictions for the three conditions given in the original
data: parametric gain, parametric loss, and distance from indifference.

It also generate three predicted BOLD signals (hemodynamic prediction) and their
time plots for each condition.
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.append("code/utils")
from hrf import *
from make_class import *

# Define some parameters
time_res, TR_subdivs = 2, 100
step_size = time_res / TR_subdivs


# Create collection of all subject IDs and all run IDs
subjects = [str(i).zfill(3) for i in range(1, 17)]
runs = [str(i).zfill(3) for i in range(1, 4)]


# We perform the procedure outlined in this script each run of each subject:
for subject in subjects:
    for run in runs:


        # Extract all relevant data stored within the ds005 files. Note that
        # the raw and filtered datasets are equal in the dimension of time.
        # We can therefore simply consider one:
        obj = ds005(subject, run)
        time_len = obj.raw.data.shape[3]


        # Create neural time courses for each regressor of interest
        time_course_gain = obj.time_course("gain", step_size)
        time_course_loss = obj.time_course("loss", step_size)
        time_course_dist2indiff = obj.time_course("dist2indiff", step_size)


        # Compute the predicted hemodynamic response function signals for
        # each regressor of interest
        neural_gain = obj.convolution("gain", step_size)
        neural_loss = obj.convolution("loss", step_size)
        neural_dist2indiff = obj.convolution("dist2indiff", step_size)


        # Create figures plotting the predicted HRF signal on time:
        time = np.arange(0, time_res * time_len, step_size)

        plt.figure(figsize=(15, 10), dpi=100)
        plt.subplot(231)
        plt.plot(time, neural_gain)
        plt.xlabel("Time")
        plt.ylabel("Hemodynamic Response")
        plt.title("Condition 1: Gain")

        plt.subplot(232)
        plt.plot(time, neural_loss)
        plt.xlabel("Time")
        plt.ylabel("Hemodynamic Response")
        plt.title("Condition 2: Loss")

        plt.subplot(233)
        plt.plot(time, neural_dist2indiff)
        plt.xlabel("Time")
        plt.ylabel("Hemodynamic Response")
        plt.title("Condition 3: Distance from Indifference")

        plt.subplot(234)
        plt.plot(time, neural_gain)
        plt.xlabel("Time")
        plt.ylabel("Neural Prediction")
        plt.title("Condition 1: Gain")

        plt.subplot(235)
        plt.plot(time, neural_loss)
        plt.xlabel("Time")
        plt.ylabel("Neural Prediction")
        plt.title("Condition 2: Loss")

        plt.subplot(236)
        plt.plot(time, neural_dist2indiff)
        plt.xlabel("Time")
        plt.ylabel("Neural Prediction")
        plt.title("Condition 3: Distance from Indifference")


        # Save these figures to the results directory
        path_result = "results/sub%s_run%s/" % (subject, run)
        bash_command = "mkdir -p " + path_result
        os.system(bash_command)

        plt.savefig(path_result + "convolution.png", dpi=500)
        plt.close()


        # Save txt files to results directory
        np.savetxt(path_result + "conv_gain.txt", neural_gain)
        np.savetxt(path_result + "conv_loss.txt", neural_loss)
        np.savetxt(path_result + "conv_dist2indiff.txt", neural_dist2indiff)
