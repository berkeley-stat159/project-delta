"""
Tests correlation module functionality

Tests can be run from the project main directory with:
    nosetests code/tests/test_stat_utils.py
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal
import numpy as np
import sys

sys.path.append("code/utils")
from stat_utils import *
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

def test_glm_util():

    # Define an instance of class ds005
    obj = ds005("test", "001")

    # Create a valid design matrix (the generalize linear model acts strangely
    # in the case of high dimensionality) and extract the response variable
    design_matrix = obj.design_matrix(gain=False, loss=False,
                                      euclidean_dist=False)
    response = obj.behav[:, 5]

    # Save and assess the output
    regression_coefficients, df, MRSS = glm_util(design_matrix, response)
    assert_almost_equal(regression_coefficients[0], 0.5)
    assert df == 1
    assert MRSS == 0.5
    