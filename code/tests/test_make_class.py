"""
Tests class `run` functionality in make_class.py

Run with:
    nosetests test_make_class.py
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_almost_equal
from .. import make_class
import numpy as np
import os

def test_make_class():

    subtest_runtest1 = make_class.run("test", "test", rm_nonresp=True)
    subtest_runtest2 = make_class.run("test", "test", rm_nonresp=False)

    # Test attribute .data
    assert subtest_runtest1.data == subtest_runtest2.data
    data = subtest_runtest1.data
    assert data.shape == (3, 3, 3, 3)
    assert [data.min(), data.max(), data.mean()] == [0, 11, 3.0]
    assert (data[..., 2] - data[..., 1] == data[..., 1] - data[..., 0]).all()

    # Test attribute .behav
    behav1, behav2 = subtest_runtest1.behav, subtest_runtest2.behav
    assert [behav1.shape, behav.shape2] == [(2, 5), (3, 5)]
    assert [behav1.min(), behav1.max(), behav1.sum()] == [0, 30, 89]
    assert [behav2.min(), behav2.max(), behav2.sum()] == [-1, 30, 129]

    # Test method .design_matrix()
    design_matrix1 = subtest_runtest1.design_matrix(resp=True)
    design_matrix2 = subtest_runtest2.design_matrix(resp_bin=True)
    assert design_matrix1.shape, design_matrix2.shape == (2, 5), (3, 5)
    assert_almost_equal(design_matrix1[0, 4] ** 2, 112.5)
    assert_almost_equal(design_matrix1[1, 4] ** 2, 12.5)
    assert_almost_equal(design_matrix2[0, 4] ** 2, 112.5)
    assert_almost_equal(design_matrix2[1, 4] ** 2, 50)
    assert_almost_equal(design_matrix2[2, 4] ** 2, 12.5)