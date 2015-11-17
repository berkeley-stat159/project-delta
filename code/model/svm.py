from __future__ import absolute_import, division, print_function
import numpy as np


def load_behav(filename, TR = 2.0):
    """
    Loads in clean data from `behavdata.txt`.

    Parameters
    ----------
    filename : str

    TR : float

    Returns
    -------
    Numpy array with columns
    """
    raw = [row.split() for row in open(filename, "r").readlines()[1:]]
    float_T = list(np.array(raw, dtype=float).T)
    onset, gain, loss, decision = float_T[0], float_T[1], float_T[2], float_T[4]
    data = np.array([onset // TR, gain, loss, decision], dtype=int).T
    return data


def svm(data, behav):
    """
    Estimates coefficients for 

    Parameters
    ----------
    data : np.ndarray
        asdf
    behav : np.ndarray
        
    TR : float
        The time it takes for the fMRI to fully capture a single volume in time

    Returns
    -------
    Numpy array, with shape[3] = 3. The first of these elements is a subarray of
    estimated coefficients for each voxel over time given `gain`. The second and
    third are the analogs given `loss` and `sentiment`, respectively.
    """