"""
Tests test data creation in make_test_data.py and class `run` in make_class.py

Run with:
    nosetests test_code.py
"""
from __future__ import absolute_import, division, print_function
import nibabel as nib
import numpy as np
import make_class, make_test_data, os
import os

def test_make_test_data():
    # Paths to directories that should contain the test subject's data
    path_data = "../data/ds005/subtest/"
    path_BOLD = path_data + "BOLD/task001_runtest/bold.nii.gz"
    path_behav = path_data + "behav/task001_runtest/behavdata.txt"

    # Test existance of test data files
    assert os.path.isfile(path_BOLD)
    assert os.path.isfile(path_behav)

    # Test BOLD data
    img = nib.load(path_BOLD)
    assert img.affine == np.eyes(4)
    data = img.get_data()
    assert data.shape == (3, 3, 3, 3)
    assert [data.min(), data.max(), data.mean()] == [0, 11, 3.0]
    assert (data[..., 2] - dat[..., 1] ==  data[..., 1] - data[..., 0]).all()

    # Test behavioral data
    behav = open(path_behav).readlines()
    assert len(behav) == 4
    assert [len(row) for row in behav] == [41, 27, 27, 26]
    assert row[0] == "onset\tgain\tloss\tPTval\trespnum\trespcat\tRT\n"
    assert row[1] == "0.00\t10\t20\t-9.80\t4\t0\t1.077\n"
    assert row[2] == "2.00\t20\t20\t0.20\t0\t-1\t0.000\n"
    assert row[3] == "4.00\t30\t20\t10.20\t2\t1\t1.328"

def test_make_class():

    subtest_runtest1 = make_class.run("test", "test", binary_resp=False)
    subtest_runtest2 = make_class.run("test", "test", rm_nonresp=False)

    # Test attribute .data
    assert subtest_runtest1.data == subtest_runtest2.data
    data = subtest_runtest1.data
    assert data.shape == (3, 3, 3, 3)
    assert [data.min(), data.max(), data.mean()] == [0, 11, 3.0]
    assert (data[..., 2] - data[..., 1] == data[..., 1] - data[..., 0]).all()

    #Test attribute .behav
    behav1, behav2 = subtest_runtest1.behav, subtest_runtest2.behav
    assert [behav1.shape, behav.shape2] == [(2, 4), (3, 4)]
    assert [behav1.min(), behav1.max(), behav1.mean()] == [0, 30, 10.375]
    assert [behav2.min(), behav2.max(), behav2.mean()] == [0, 30, 10.75]