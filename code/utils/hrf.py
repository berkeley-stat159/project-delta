"""
This script contains the hrf() function, which is a utility for computing values
of the hemodynamic response function at given times. This is a necessary tool
for predicting BOLD signals. Future Python scripts can take advantage of the
hrf() function by including the command
    sys.path.append("code/utils")
    from hrf import *
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.stats import gamma

def hrf(times):
    """
    Computes values for the canonical hemodynamic response function at the
    specified times 
    
    Parameters
    ----------
    times : np.ndarray
        1-D array of time points

    Return:
    ------
    hrf : np.ndarray
        Array of shape (len(times),) that represents the hemodynamic response
        function for the specified time points
    """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6
    