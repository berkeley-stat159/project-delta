"""
==================Test file for convolution.py======================
Test convolution module, hrf function and convolve function

Run with:
		nosetests test_convolution.py

"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.stats import gamma
from stimuli import events2neural
from convolution import *

def test_hrf():
	times = np.arange(30)
	hr = hrf(times)
	assert len(times)==len(hr)

def test_convolve():
	TR = 2
	n_vols = 240
	duration = 1.5
	neural = events2neural(".././cond001.txt", TR, n_vols)
	conv = convolve(neural, TR, n_vols, 1.5)
	assert len(conv)==n_vols

