from nose.tools import assert_equals

import corr_per_voxel

data = './temp_data_for_testing/bold.nii.gz'
txst = './temp_data_for_testing/cond001.txt'
def test_foo_10():
    corr_act = corr_per_voxel.corr(data,txst)
    assert_equals(corr_act, 100)


