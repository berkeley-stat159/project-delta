"""
Tests creation of test data in make_test_data.py

Run with:
    nosetests test_make_test_data.py
"""
from __future__ import absolute_import, division, print_function
import nibabel as nib
import numpy as np
import os, sys

sys.path.insert(0, "code/utils")
import make_test_data

def test_make_test_data():
    # Paths to directories that should contain the test subject's data
    path_data = "data/ds005/subtest1/"
    path_BOLD = path_data + "BOLD/task001_run001/bold.nii.gz"
    path_behav = path_data + "behav/task001_run001/behavdata.txt"

    # Test existance of test data files
    assert os.path.isfile(path_BOLD)
    assert os.path.isfile(path_behav)

    # Test BOLD data
    img = nib.load(path_BOLD)
    assert (img.affine == np.eye(4)).all()
    data = img.get_data()
    assert data.shape == (3, 3, 3, 3)
    assert [data.min(), data.max(), data.mean()] == [0, 11, 3.0]
    assert (data[..., 2] - data[..., 1] ==  data[..., 1] - data[..., 0]).all()

    # Test behavioral data
    behav = open(path_behav).readlines()
    assert len(behav) == 4
    assert [len(row) for row in behav] == [41, 27, 27, 26]
    assert behav[0] == "onset\tgain\tloss\tPTval\trespnum\trespcat\tRT\n"
    assert behav[1] == "0.00\t10\t20\t-9.80\t4\t0\t1.077\n"
    assert behav[2] == "2.00\t20\t20\t0.20\t0\t-1\t0.000\n"
    assert behav[3] == "4.00\t30\t20\t10.20\t2\t1\t1.328"
