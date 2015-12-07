"""
Tests class `run` functionality in make_analysis.py

Tests can be run from the project main directory with:
    nosetests code/tests/test_make_analysis.py

in the main project directory
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_equal
import numpy as np
import os, sys

sys.path.append("code/utils")
from convolution import hrf
import make_analysis

def test_make_class():
    

    # Test method .behav_analysis()
    sub = make_class.run("test", "001", filtered_data=True)
    behav_beta1, behav_lam1, misclass_rate1, p1 = sub.behav_analysis(euclidean_dist=True)
    behav_beta2, behav_lam2, misclass_rate2, p2 = sub.behav_analysis(euclidean_dist=False)

    assert np.all(misclass_rate1 >= 0) 
    assert np.all(misclass_rate2 <= 1)
    assert np.all(p1 >= 0) 
    assert np.all(p1 <= 1)
 
    # Test bold_predict()
    sub.bold_predict()

    # Test attribute .hemo_pred_gain 
    n_vols = sub.data.shape[-1]
    assert_equal(len(sub.hemo_pred_gain, n_vols)
    # Test attribute .hemo_pred_loss
    assert_equal(len(sub.hemo_pred_loss, n_vols)
    # Test attribute .hemo_pred_dist   
    assert_equal(len(sub.hemo_pred_dist, n_vols)


    # Test method .bold_figure()
    sub.bold_figure()

    # Test method .outlier_detection()
    indices = sub.outlier_detection()
    rmsd = vol_rms_diff(sub.data)
    rmsd_outlier_id = iqr_outliers(rmsd)[0]
    assert_array_equal(diagnostics.extend_diff_outliers(rmsd_outlier_id),
                       indices)


    # Test method .linear_analysis(self, rm_outlier=False)


