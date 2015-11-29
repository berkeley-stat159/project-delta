"""
Tests pearson_1d and pearson_2d functions in pearson.py module

Tests can be run from project main directory with:
    nosetests code/utils/tests/test_pearson.py
"""
from __future__ import print_function, division, absolute_import
from numpy.testing import assert_almost_equal
import numpy as np
import sys

sys.path.append("code/utils")
import pearson

def test_pearson_1d():
    # Test pearson_1d routine
    x = np.random.rand(22)
    y = np.random.rand(22)
    # Does routine give same answer as np.corrcoef?
    expected = np.corrcoef(x, y)[0, 1]
    actual = pearson_1d(x, y)
    # Did you, gentle user, forget to return the value?
    if actual is None: raise RuntimeError("function returned None")
    assert_almost_equal(expected, actual)

def test_pearson_2d():
    # Test pearson_2d routine
    x = np.random.rand(22)
    Y = np.random.normal(size=(22, 12))
    # Does routine give same answers as np.corrcoef?
    expected = np.corrcoef(x, Y.T)[0, 1:]
    actual = pearson_2d(x, Y)
    # Did you, gentle user, forget to return the value?
    if actual is None: raise RuntimeError("function returned None")
    assert_almost_equal(expected, actual)
