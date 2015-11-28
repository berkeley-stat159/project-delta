""" Functions to work with standard OpenFMRI stimulus files

The functions have docstrings according to the numpy docstring standard - see:

    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
"""
from __future__ import absolute_import, division, print_function
import numpy as np

def events2neural(task_fname, tr, n_trs):
    """ Return predicted neural time course from event file `task_fname`

    Parameters
    ----------
    task_fname : str
        Filename of event file
    tr : float
        TR in seconds
    n_trs : int
        Number of TRs in functional run

    Returns
    -------
    time_course : array shape (n_trs,)
        Predicted neural time course, one value per TR
    """
    task = np.loadtxt(task_fname)
    # Check that the file is plausibly a task file
    if task.ndim != 2 or task.shape[1] != 3:
        raise ValueError("Is {0} really a task file?", task_fname)
    # Convert onset, duration seconds to TRs
    task[:, :2] = task[:, :2] / tr
    # Neural time course from onset, duration, amplitude for each event
    time_course = np.zeros(n_trs)
    for onset, duration, amplitude in task:
        time_course[onset:onset + duration] = amplitude
    return time_course

def neural_highres(onsets, durations, amplitudes, TR=2, n_trs=240, tr_divs=100):
    """Return predicted neural time course in the case when onsets are not equally spaced and do not start on a TR.

    Parameters:
    ----------
    onsets: an array
        the time points when a task starts
    duration: an array
        the duration of a task
    amplitudes: an array
        amplitudes corresponding to each task
    tr_divs: int (default value is 10)
        step size per TR
    n_trs: int (default: 240)
        number of TRs 
    TR: float (default: 2)
        time to repetition

    Return
    -------
    hr_neural: an array (n_trs*tr_divs, )
        predicted neural time course
    """
    onsets_in_scans = onsets/TR
    high_res_times = np.arange(0, n_trs, 1/tr_divs)*TR
    high_res_neural = np.zeros(high_res_times.shape)
    high_res_onset_indices = onsets_in_scans * tr_divs
    high_res_durations = durations/TR*tr_divs
    for hr_onset, hr_duration, amplitude in list(zip(high_res_onset_indices,high_res_durations,amplitudes)):
        hr_onset = int(round(hr_onset))
        hr_duration = int(round(hr_duration))
        high_res_neural[hr_onset:hr_onset+hr_duration] = amplitude
    return high_res_neural

