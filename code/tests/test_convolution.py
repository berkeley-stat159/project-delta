"""
==================Test file for convolution.py======================
Test convolution module: hrf function and convolve function

Run with:
        nosetests code/tests/test_convolution.py

"""
from __future__ import absolute_import, division, print_function
import numpy as np
from nose.tools import assert_equal
from scipy.stats import gamma
import sys
sys.path.append("code/utils")
from make_class import *
from convolution import *

TR = 2
sub = run("001","001")

def test_hrf():
    times = np.arange(30)
    hr = hrf(times)
    assert_equal(len(times), len(hr))

def test_convolve():
    n_vols = sub.data.shape[-1]
    neural = sub.time_course("gain")
    conv = convolve(neural, TR, n_vols)
    assert_equal(len(conv), n_vols)

