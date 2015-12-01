"""
Testing plotting functions
"""
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("code/utils")
from plot_tool import *

data = np.arange(512)
data.shape = (8,8,8)

def test_show_slice():
    a = show_slice(data, 6,color_option="bwr")
    assert a is None

def test_show3Dimage():
    a = show3Dimage(data,color_option="seismic")
    assert a is None