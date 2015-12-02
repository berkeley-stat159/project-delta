"""
Testing plotting functions
"""
from __future__ import division, print_function, absolute_import
from nose.tools import assert_equal
import numpy as np
import sys
sys.path.append("code/utils")
from plot_tool import *

def test_show3Dimage():
    data3d = np.arange(512)
    data3d.shape = (8,8,8)
    data4d = np.arange(256)
    data4d.shape = (4,4,4,4)
    a = show3Dimage(data3d, time=None)
    b = show3Dimage(data4d, time=2)
    assert_equal(a.shape, (24, 24))
    assert_equal(b.shape, (8, 8))