"""
Tests run() class functionality in make_class module

Tests can be run from the project main directory with:
    nosetests code/tests/test_make_class.py
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal
import numpy as np
import sys

sys.path.append("code/utils")
from make_class import *


def test_img():
    
    # Import test data from raw and filtered data files
    assert_raises(AssertionError, img, "data/ds005/sub000/BOLD/" +
                  "task001_run000/bold.nii.gz")
    image = img("data/ds005/subtest/BOLD/task001_run001/bold.nii.gz")
    data, affine, voxels_per_mm = image.data, image.affine, image.voxels_per_mm

    # Test attribute .data
    assert data.shape == (3, 3, 3, 3)
    assert [data.min(), data.max(), data.mean()] == [0, 11, 3]
    assert_array_equal(data[..., 2] - data[..., 1], data[..., 1] - data[..., 0])

    # Test attribute .affine
    assert_array_equal(affine, np.eye(4))

    # Test attribute .voxels_per_mm
    assert_array_equal(voxels_per_mm, np.array([1, 1, 1, 0]))

    # Test method .smooth()
    assert_raises(AssertionError, image.smooth, np.array([5, 5, 5]))
    assert_raises(AssertionError, image.smooth, np.array(["5", "5", "5", "5"]))
    assert_raises(AssertionError, image.smooth, "five")
    smooth = image.smooth()
    assert smooth.shape == (3, 3, 3, 3)
    assert [smooth.min(), smooth.max(), smooth.sum()] == [0, 5, 108]


def test_ds005():
    
    # Save expected results to global environment
    image = img("data/ds005/subtest/BOLD/task001_run001/bold.nii.gz")
    data, affine, voxels_per_mm = image.data, image.affine, image.voxels_per_mm
    smooth = image.smooth()

    # Save test data and expected results to global environment
    ds005_1 = ds005("test", "001")
    ds005_2 = ds005("test", "001", rm_nonresp=False)

    # Test consistency of .raw attributes and methods
    assert_array_equal(ds005_1.raw.data, ds005_2.raw.data)
    assert_array_equal(ds005_1.raw.affine, ds005_2.raw.affine)
    assert_array_equal(ds005_1.raw.voxels_per_mm, ds005_2.raw.voxels_per_mm)
    assert_array_equal(ds005_1.raw.smooth(), ds005_2.raw.smooth())

    # Test accuracy of .raw attributes and methods
    assert_array_equal(ds005_1.raw.data, data)
    assert_array_equal(ds005_1.raw.affine, affine)
    assert_array_equal(ds005_1.raw.voxels_per_mm, voxels_per_mm)
    assert_array_equal(ds005_1.raw.smooth(), smooth)

    # Test consistency of .filtered attributes and methods
    assert_array_equal(ds005_1.filtered.data, ds005_2.filtered.data)
    assert_array_equal(ds005_1.filtered.affine, ds005_2.filtered.affine)
    assert_array_equal(ds005_1.filtered.voxels_per_mm,
                       ds005_2.filtered.voxels_per_mm)
    assert_array_equal(ds005_1.filtered.smooth(), ds005_2.filtered.smooth())

    # Test accuracy of .filtered attributes and methods
    assert_array_equal(ds005_1.filtered.data, data)
    assert_array_equal(ds005_1.filtered.affine, affine)
    assert_array_equal(ds005_1.filtered.voxels_per_mm, voxels_per_mm)
    assert_array_equal(ds005_1.filtered.smooth(), smooth)

    # Test .behav attribute
    assert [ds005_1.behav.shape, ds005_2.behav.shape] == [(2, 7), (3, 7)]
    assert_array_equal(ds005_1.behav[:, 0], np.array([0, 4]))
    assert_almost_equal((ds005_1.behav[:, 3] ** 2).sum(), 125)
    assert_array_equal(ds005_2.behav[:, 0], np.array([0, 2, 4]))
    assert_almost_equal((ds005_2.behav[:, 3] ** 2).sum(), 175)

    # Test .cond_* attributes
    assert_array_equal(ds005_1.cond_gain, ds005_2.cond_gain)
    assert_array_equal(ds005_1.cond_loss, ds005_2.cond_loss)
    assert_array_equal(ds005_1.cond_dist2indiff, ds005_2.cond_dist2indiff)
    assert_array_equal(ds005_1.cond_gain, ds005_1.cond_dist2indiff)
    assert_array_equal(ds005_1.cond_gain[:, 0], ds005_1.cond_loss[:, 0])
    assert_array_equal(ds005_1.cond_gain[:, 1], ds005_1.cond_loss[:, 1])
    assert_array_equal(ds005_1.cond_gain[:, 0], np.array([0, 2, 4]))
    assert_array_equal(ds005_1.cond_gain[:, 1], np.array([1, 1, 1]))
    assert_array_equal(ds005_1.cond_gain[:, 2], np.array([-1, 0, 1]))
    assert_array_equal(ds005_1.cond_loss[:, 2], np.array([0, 0, 0]))

    # Test .design_matrix() method
    assert ds005_1.design_matrix().shape == (2, 4)
    assert ds005_1.design_matrix(gain=False, loss=False).shape == (2, 2)
    assert ds005_1.design_matrix(euclidean_dist=False).shape == (2, 3)
    assert ds005_1.design_matrix(resp_time=True).shape == (2, 5)
    assert_almost_equal(ds005_1.design_matrix().sum(), 96.142135623730951)
    assert ds005_2.design_matrix(gain=False, loss=False).shape == (3, 2)
    assert ds005_2.design_matrix(euclidean_dist=False).shape == (3, 3)
    assert ds005_2.design_matrix(resp_time=True).shape == (3, 5)
    assert_almost_equal(ds005_2.design_matrix().sum(), 144.21320343559643)

    # Test method .time_course()
    assert_raises(AssertionError, ds005_1.time_course, "GAIN")
    time_course = ds005_1.time_course("gain")
    assert_array_equal(time_course, ds005_2.time_course("gain"))
    assert_array_equal(time_course, np.array([-1, 0, 1]))

    # Test method .convolution()
    convolution = ds005_1.convolution("loss")
    assert_array_equal(convolution, ds005_2.convolution("loss"))
    assert_array_equal(convolution, np.array([0, 0, 0]))
    