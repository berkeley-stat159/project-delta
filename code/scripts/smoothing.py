"""
Purpose
-------
This script applies smoothing with a Gaussian kernel in bulk to both the raw and
the filtered BOLD signal data. The kernel used has a full-width-at-half-maximum
measurement of 5 millimeters, which corresponds to a standard deviation of
approximately 2.355 millimeters.

It should output four files per run: two each for the raw and the filtered data,
one of which is the image before smoothing and the other the image after
smoothing.
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.append("code/utils")
from make_class import *
from plot_tool import *


# Create a collection of all subject IDs and all run IDs
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
IDs = list(zip([run_ID for _ in range(16) for run_ID in run_IDs],
               [subject_ID for _ in range(3) for subject_ID in subject_IDs]))
IDs.sort()

# We perform the procedure outlined in this script for each run of each subject:
for ID in IDs:
    run, subject = ID


    # Extract the data of interest
    obj = ds005(subject, run)


    # Define results directories to which to save the figures produced
    path_result = "results/sub%s_run%s/smoothing/" % (subject, run)
    try:
        os.makedirs(path_result)
    except OSError:
        if not os.path.isdir(path_result):
            raise

    # Each figure will be plotted with the help of the plot_volume() utility
    # contained in the plot_tool module. It produces a two-dimensional grid-like
    # canvas on which each horizontal slice of the brain is shown as a tile.
    raw_original = plot_volume(obj.raw.data, 50)
    plt.imshow(raw_original)
    plt.colorbar()
    plt.title("Raw Data: Before Smoothing")
    plt.savefig(path_result + "raw_original.png")
    plt.close()

    raw_smoothed = plot_volume(obj.raw.smooth(), 50)
    plt.imshow(raw_smoothed)
    plt.colorbar()
    plt.title("Raw Data: After Smoothing")
    plt.savefig(path_result + "raw_smoothed.png")
    plt.close()

    filtered_original = plot_volume(obj.filtered.data, 50)
    plt.imshow(filtered_original)
    plt.colorbar()
    plt.title("Filtered Data: Before Smoothing")
    plt.savefig(path_result + "filtered_original.png")
    plt.close()

    filtered_smoothed = plot_volume(obj.filtered.smooth(), 50)
    plt.imshow(filtered_smoothed)
    plt.colorbar()
    plt.title("Filtered Data: After Smoothing")
    plt.savefig(path_result + "filtered_smoothed.png")
    plt.close()
