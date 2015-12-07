"""
Convolving predicted neural time course with the hemodynamic response function
Getting predicted BOLD signals

Notes:
Since the duration of the task is 3 seconds and the onsets are not equally spaced, we have to
mannually calcualte convolution instead of using builtin np.convolve

"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.stats import gamma

def hrf(times):
    """
    Computes values for the canonical hemodynamic response function at specified
    times 
    
    Parameters
    ----------
    times: np.ndarray
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
    