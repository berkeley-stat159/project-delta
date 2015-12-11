"""
Tests validity of test data in make_test_data.py

Tests can be run from project main directory with:
    nosetests code/tests/test_make_test_data.py
"""
from __future__ import absolute_import, division, print_function
import nibabel as nib
import numpy as np
import os, sys

sys.path.append("code/utils")
import make_test_data

def test_make_test_data():
    # Paths to select files contain test subject data
    path_data = "data/ds005/subtest/"
    files = [path_data + "BOLD/task001_run001/bold.nii.gz",
             path_data + ("model/model001/task001_run001.feat/" +
                          "filtered_func_data_mni.nii.gz"),
             path_data + "behav/task001_run001/behavdata.txt",
             path_data + "model/model001/onsets/task001_run001/cond002.txt"]

    # First check existence of test files
    for file in files: assert os.path.isfile(file)

    # Test BOLD data
    img = nib.load(files[0])
    assert (img.affine == np.eye(4)).all()
    data = img.get_data()
    assert data.shape == (3, 3, 3, 3)
    assert [data.min(), data.max(), data.mean()] == [0, 11, 3.0]
    assert (data[..., 2] - data[..., 1] ==  data[..., 1] - data[..., 0]).all()

    # Test filtered BOLD data
    img = nib.load(files[1])
    assert (img.affine == np.eye(4)).all()
    data = img.get_data()
    assert data.shape == (3, 3, 3, 3)
    assert [data.min(), data.max(), data.mean()] == [0, 11, 3.0]
    assert (data[..., 2] - data[..., 1] ==  data[..., 1] - data[..., 0]).all()

    # Test behavioral data
    behav = open(files[2]).readlines()
    assert behav[0] == "onset\tgain\tloss\tPTval\trespnum\trespcat\tRT\n"
    assert behav[1] == "0.00\t10\t20\t-9.80\t4\t0\t1.077\n"
    assert behav[2] == "2.00\t20\t20\t0.20\t0\t-1\t0.000\n"
    assert behav[3] == "4.00\t30\t20\t10.20\t2\t1\t1.328\n"

    # Test task condition data
    cond = open(files[3]).readlines()
    assert cond[0] == "0.0000\t1\t-1.0000\n"
    assert cond[1] == "2.0000\t1\t0.0000\n"
    assert cond[2] == "4.0000\t1\t1.0000\n"
