"""
Tests class `run` in make_class.py

Run with:
    nosetests test_make_class.py
"""
from __future__ import absolute_import, division, print_function
import nibabel as nib
import numpy as np
import make_class

subtest_runtest1 = make_class.run("test", "test", binary_resp=False)
subtest_runtest2 = make_class.run("test", "test", rm_nonresp=False)

def test_run():
    assert subtest_runtest1.data == subtest_runtest2.data
    data = subtest_runtest1.data
    assert data.shape == (3, 3, 3, 3)
    assert [data.min(), data.max(), data.mean()] == [0, 11, 3.0]
    assert (data[..., 2] - data[..., 1] == data[..., 1] - data[..., 0]).all()
    behav1, behav2 = subtest_runtest1.behav, subtest_runtest2.behav
    assert [behav1.shape, behav.shape2] == [(2, 4), (3, 4)]
    assert [behav1.min(), behav1.max(), behav1.mean()] == [0, 30, 10.375]
    assert [behav2.min(), behav2.max(), behav2.mean()] == [0, 30, 10.75]