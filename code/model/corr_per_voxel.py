import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from stimuli import events2neural

img = nib.load('./temp_data_for_testing/bold.nii.gz')
n_trs = img.shape[-1]
TR = 2

# Call the events2neural function to generate the on-off values for each volume
time_course = events2neural('./temp_data_for_testing/cond001.txt', 2, n_trs)
data = img.get_data()
data = data[..., 4:]
time_course = time_course[4:]
n_voxels = np.prod(data.shape[:-1])
data_2d = np.reshape(data, (n_voxels, data.shape[-1]))
correlations_1d = np.zeros((n_voxels,))
for i in range(n_voxels):
    correlations_1d[i] = np.corrcoef(time_course, data_2d[i, :])[0, 1]
correlations = np.reshape(correlations_1d, data.shape[:-1])

# Plot the middle slice of the third axis from the correlations array
plt.imshow(correlations[:, :, 14])



