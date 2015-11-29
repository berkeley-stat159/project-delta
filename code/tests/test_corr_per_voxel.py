from __future__ import division, print_function, absolute_import
from nose.tools import assert_equals
import nibabel as nib
import numpy as np
import sys

sys.path.append("code/utils")
from stimuli import events2neural

sys.path.append("code/model")
from corr_per_voxel import *

nii_file = 'bold.nii.gz'
cond_file = 'cond001.txt'
def test_corr():
	corr_act = corr(nii_file,cond_file)
	img = nib.load(nii_file)
	n_trs = img.shape[-1]
	TR = 2
	time_course = events2neural(cond_file, 2, n_trs)
	data = img.get_data() 
	data = data[..., 4:]
	time_course = time_course[4:]
	n_voxels = np.prod(data.shape[:-1]) 
	assert corr_act.shape == data.shape[:-1]

