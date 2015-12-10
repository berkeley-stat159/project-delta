"""
Testing plotting utilities contained in plot_tool module

Tests can be run from project main directory with
    nosetests code/tests/test_plot_tool
"""
from __future__ import division, print_function, absolute_import
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
import numpy as np
import sys

sys.path.append("code/utils")
from plot_tool import *

def test_plot_volume():

    # Import test data
    data_file = "data/ds005/subtest/BOLD/task001_run001/bold.nii.gz"
    data4d = nib.load(data_file).get_data()
    data3d = data4d[..., 0]

    # Tests argument volume functionality
    assert_raises(AssertionError, plot_volume, 2)
    assert_raises(AssertionError, plot_volume, data4d)
    assert_raises(AssertionError, plot_volume, np.array([1]))

    # Define test data sets
    canvas1 = plot_volume(data3d)
    canvas2 = plot_volume(data4d, 0)
    canvas3 = plot_volume(data4d, 1)
    canvas4 = plot_volume(data4d, 2)
    canvases = [canvas1, canvas2, canvas3, canvas4]
    for canvas in canvases: canvas[np.isnan(canvas)] = 0

    # Check shapes and contents
    for canvas in canvases: assert canvas.shape == (6, 6)
    for canvas in canvases: assert_array_equal(canvas[3:, 3:], np.zeros((3, 3)))
    assert_array_equal(canvas1, canvas2)
    assert canvas1.sum() == data3d.sum()
    assert canvas3.sum() - canvas2.sum() == 54
    assert canvas4.sum() - canvas3.sum() == 54
