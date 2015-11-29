"""
Tests functions in diagnostics.py

Run with:
    nosetests test_diagnostics.py
"""

from __future__ import absolute_import, division, print_function
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal
import numpy as np

sys.path.append("code/model")
import diagnostics as diag

def test_vol_std():
    # We make a fake 4D image
    shape_3d = (2, 3, 4)
    V = np.prod(shape_3d)
    T = 10  # The number of 3D volumes
    # Make a 2D array that we will reshape to 4D
    arr_2d = np.random.normal(size=(V, T))
    expected_stds = np.std(arr_2d, axis=0)
    # Reshape to 4D
    arr_4d = np.reshape(arr_2d, shape_3d + (T,))
    actual_stds = diag.vol_std(arr_4d)
    assert_almost_equal(expected_stds, actual_stds)

def test_iqr_outliers():
    # Test with simplest possible array
    arr = np.arange(101)  # percentile same as value
    # iqr = 50
    exp_lo = 25 - 75
    exp_hi = 75 + 75
    indices, thresholds = diag,iqr_outliers(arr)
    assert_array_equal(indices, [])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Reverse, same values
    indices, thresholds = diag.iqr_outliers(arr[::-1])
    assert_array_equal(indices, [])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Add outliers
    arr[0] = -51
    arr[1] = 151
    arr[100] = 1  # replace lost value to keep centiles same
    indices, thresholds = diag.iqr_outliers(arr)
    assert_array_equal(indices, [0, 1])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Reversed, then the indices are reversed
    indices, thresholds = diag.iqr_outliers(arr[::-1])
    assert_array_equal(indices, [99, 100])
    assert_equal(thresholds, (exp_lo, exp_hi))

def test_iqr_scaling():
    # Check that the scaling of IQR works
    # Test with simplest possible array
    arr = np.arange(101)  # percentile same as value
    # iqr = 50
    exp_lo = 25 - 100
    exp_hi = 75 + 100
    indices, thresholds = diag.iqr_outliers(arr, 2)
    assert_array_equal(indices, [])
    assert_equal(thresholds, (exp_lo, exp_hi))
    # Add outliers - but these aren't big enough now
    arr[0] = -51
    arr[1] = 151
    indices, thresholds = diag.iqr_outliers(arr, 2)
    assert_array_equal(indices, [])
    # Add outliers - that are big enough
    arr[0] = -76
    arr[1] = 176
    arr[100] = 1  # replace lost value to keep centiles same
    indices, thresholds = diag.iqr_outliers(arr, 2)
    assert_array_equal(indices, [0, 1])

def test_vol_rms_diff():
    # We make a fake 4D image
    shape_3d = (2, 3, 4)
    V = np.prod(shape_3d)
    T = 10  # The number of 3D volumes
    # Make a 2D array that we will reshape to 4D
    arr_2d = np.random.normal(size=(V, T))
    differences = np.diff(arr_2d, axis=1)
    exp_rms = np.sqrt(np.mean(differences ** 2, axis=0))
    # Reshape to 4D and run function
    arr_4d = np.reshape(arr_2d, shape_3d + (T,))
    actual_rms = diag.vol_rms_diff(arr_4d)
    assert_almost_equal(actual_rms, exp_rms)

def test_extend_diff_outliers():
    # Test function to extend difference outlier indices
    indices = np.array([3, 7, 12, 20])
    assert_array_equal(diag.extend_diff_outliers(indices),
                       [3, 4, 7, 8, 12, 13, 20, 21])

def test_sequential_input():
    indices = np.array([4, 5, 9, 10])
    assert_array_equal(diag.extend_diff_outliers(indices),
                       [4, 5, 6, 9, 10, 11])
    indices = np.array([1, 2, 4, 5, 9, 10])
    assert_array_equal(diag.extend_diff_outliers(indices),
                       [1, 2, 3, 4, 5, 6, 9, 10, 11])
    indices = np.array([3, 7, 8, 12, 20])
    assert_array_equal(diag.extend_diff_outliers(indices),
                       [3, 4, 7, 8, 9, 12, 13, 20, 21])