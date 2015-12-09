"""
Tests correlation module functionality

Tests can be run from the project main directory with:
    nosetests code/tests/test_correlation.py
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
import numpy as np
import sys

sys.path.append("code/utils")
from correlation import *
from make_class import *

def test_correlation():

    # Define an instance of class ds005
    obj = ds005("test", "001")

    # Test proper raising of AssertionErrors
    assert_raises(AssertionError, correlation, obj.raw, "raw", "gain")
    assert_raises(AssertionError, correlation, obj, "Raw", "gain")

    # Test correlation output
    corr = correlation(obj, "raw", "gain")
    expected = np.ones((3, 3, 3))
    assert_array_equal(corr, expected)
    