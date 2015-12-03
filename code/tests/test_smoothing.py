"""
==================Test file for smoothing.py======================

Test convolution module, hrf function and convolve function

Run with:
		nosetests nosetests code/tests/test_smoothing.py


"""

from __future__ import absolute_import, division, print_function
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal
import numpy as np
import sys

sys.path.append("code/utils")
from smoothing import *
import make_class

subtest_runtest1 = make_class.run("test", "001", filtered_data=True)

# Test method .smooth()
smooth1, smooth2 = subtest_runtest1.smooth(0), subtest_runtest1.smooth(1, 5)
smooth3 = subtest_runtest1.smooth(2, 0.25)
assert [smooth1.max(), smooth1.shape, smooth1.sum()] == [0, (3, 3, 3), 0]
assert [smooth2.max(), smooth2.shape, smooth2.sum()] == [1, (3, 3, 3), 27]
assert [smooth3.max(), smooth3.shape, smooth3.sum()] == [8, (3, 3, 3), 108]
assert [smooth1.std(), smooth2.std()] == [0, 0]
assert_almost_equal(smooth3.std(), 1.6329931618554521)