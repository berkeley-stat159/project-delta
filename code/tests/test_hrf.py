"""
Tests hrf() functionality in hrf module

Tests can be run from the project main directory with:
    nosetests code/tests/test_convolution.py
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from nose.tools import assert_equal
from scipy.stats import gamma
import sys

sys.path.append("code/utils")
from hrf import *

def test_hrf():
	# Define the only two parameters of interest
    times = np.arange(30)
    hr = hrf(times)

    # They should contain the same number of elements
    assert_equal(len(times), len(hr))

    # The hemodynamic response is 0 at onset
    assert hr[0] == 0

    # HRF initially increases, then decreases, then increases/stabilizes to 0
    for i in range(0, 4): assert hr[i] < hr[i + 1]
    for i in range(5, 10): assert hr[i] > hr[i + 1]
    for i in range(15, 29): assert hr[i] < hr[i + 1]

