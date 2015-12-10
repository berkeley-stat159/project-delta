"""
Purpose
-------
This script allows production, for each run of each subject, of convolved
hemodynamic response predictions for the three conditions given in the original
data: parametric gain, parametric loss, and distance from indifference.

It should generate four files per run: one figure that contains six plots (one
depicting the hemodynamic response and one depicting the neural prediction, for
each run and condition), and one plaintext file that contains the convolved
hemodynamic response function predictions for each of the three conditions.
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.append("code/utils")
from make_class import *


# Define some parameters
time_res, TR_subdivs = 2, 100
step_size = time_res / TR_subdivs


# Create a collection of all subject IDs and all run IDs
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
IDs = list(zip([subject_ID for _ in range(3) for subject_ID in subject_IDs],
              [run_ID for _ in range(16) for run_ID in run_IDs]))


# We perform the procedure outlined in this script for each run of each subject:
for ID in IDs:
    subject, run = ID


    # Extract all relevant data stored within the ds005 files. Note that the raw
    # and filtered datasets are equal in the dimension of time. We can therefore
    # simply consider one:
    obj = ds005(subject, run)
    time_len = obj.raw.data.shape[3]


    # Create neural time courses for each regressor of interest
    time_course_gain = obj.time_course("gain", step_size)
    time_course_loss = obj.time_course("loss", step_size)
    time_course_dist2indiff = obj.time_course("dist2indiff", step_size)


    # Compute the predicted hemodynamic response function signals for each
    # regressor of interest
    neural_gain = obj.convolution("gain", step_size)
    neural_loss = obj.convolution("loss", step_size)
    neural_dist2indiff = obj.convolution("dist2indiff", step_size)


    # Create figures plotting the predicted HRF signal on time:
    time = np.arange(0, time_res * time_len, step_size)

    plt.figure(figsize=(15, 10), dpi=100)
    plt.subplot(231)
    plt.plot(time, time_course_gain)
    plt.xlabel("Time")
    plt.ylabel("Hemodynamic Response")
    plt.title("Condition 1: Gain")

    plt.subplot(232)
    plt.plot(time, time_course_loss)
    plt.xlabel("Time")
    plt.ylabel("Hemodynamic Response")
    plt.title("Condition 2: Loss")

    plt.subplot(233)
    plt.plot(time, time_course_dist2indiff)
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


    # Define results directories to which to save the findings of this analysis
    path_result = "results/sub%s_run%s/convolution/" % (subject, run)
    for path in paths:
        try:
            os.makedirs(path_result)
        except OSError:
            if not os.path.isdir(path_result):
                raise


    # Save these figures to the results directory
    plt.savefig(path_result + "convolution.png", dpi=500)
    plt.close()


    # Save txt files to results directory
    np.savetxt(path_result + "conv_gain.txt", neural_gain)
    np.savetxt(path_result + "conv_loss.txt", neural_loss)
    np.savetxt(path_result + "conv_dist2indiff.txt", neural_dist2indiff)
