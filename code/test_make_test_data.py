"""
Tests creation of test data in make_test_data.py

Run at the project directory with:
    nosetests code/test_make_test_data.py
"""
from __future__ import absolute_import, division, print_function
from numpy.testing import assert_array_equal
import nibabel as nib
import numpy as np
import os

def test_make_test_data():
    # Paths to directories that should contain the test subject's data
    sub1, sub2 = "../data/ds005/subtest1/", "../data/ds005/subtest2/"
    path_BOLD_11 = sub1 + "BOLD/task001_run001/bold.nii.gz"
    path_BOLD_12 = sub1 + "BOLD/task001_run002/bold.nii.gz"
    path_BOLD_21 = sub2 + "BOLD/task001_run001/bold.nii.gz"
    path_BOLD_22 = sub2 + "BOLD/task001_run002/bold.nii.gz"
    path_behav_11 = sub1 + "behav/task001_run001/behavdata.txt"
    path_behav_12 = sub1 + "behav/task001_run002/behavdata.txt"
    path_behav_21 = sub2 + "behav/task001_run001/behavdata.txt"
    path_behav_22 = sub2 + "behav/task001_run002/behavdata.txt"
    paths = [path_BOLD_11, path_BOLD_12, path_BOLD_21, path_BOLD_22,
         path_behav_11, path_behav_12, path_behav_21, path_behav_22]

    # Test existance of test data files
    for path in paths: assert os.path.isfile(path)

    # Test BOLD data
    for path_BOLD in paths[:4]:
        img = nib.load(path_BOLD)
        assert_array_equal(img.affine, np.eye(4))
        data = img.get_data()
        assert data.shape == (3, 3, 3, 3)
        assert [data.min(), data.max(), data.mean()] == [0, 11, 3.0]
        assert_array_equal(data[..., 2] - data[..., 1],
                           data[..., 1] - data[..., 0])

    # Test behavioral data
    for path_behav in paths[4:]:
        behav = open(path_behav).readlines()
        assert len(behav) == 4
        assert [len(row) for row in behav] == [41, 27, 27, 26]
        assert behav[0] == "onset\tgain\tloss\tPTval\trespnum\trespcat\tRT\n"
        assert behav[1] == "0.00\t10\t20\t-9.80\t4\t0\t1.077\n"
        assert behav[2] == "2.00\t20\t20\t0.20\t0\t-1\t0.000\n"
        assert behav[3] == "4.00\t30\t20\t10.20\t2\t1\t1.328"