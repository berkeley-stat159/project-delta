from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage.filters import gaussian_filter

class run(object):
    """
    This class allows organization of the data by runs. Methods attached perform
    the indicated analyses of the data.
    """

    def __init__(self, sub_id, run_id, rm_nonresp=True, time_correct=False):
        """
        Each object of this class created contains the fMRI data along with the
        corresponding behavioral data.

        Parameters
        ----------
        sub_id : str
            Unique key used to identify the subject (i.e., 001, ..., 016)
        run_id : str
            Unique key used to identify the run number (i.e, 001, ..., 003)
        rm_nonresp : bool, optional
            True removes trials that resulted in subject nonresponse
        time_correct : bool, optional
            True divides onsets by two to match indices of corresponding volumes
        """
        # Save the path to the directory containing the subject's data
        path_data = "../../data/ds005/sub%s/" % (sub_id,)

        # Extract subject's filtered_func_data_mni data for specidied run
        path_affine = "../../data/ds005_mnifunc/sub%s/model/model001/task001_run%s.feat/filtered_func_data_mni.nii.gz" % (sub_id, run_id)
        self.affine = nib.load(path_affine).affine

        # Extract subject's BOLD signal data for the specified run
        path_BOLD = path_data + "BOLD/task001_run%s/bold.nii.gz" % (run_id,)
        self.data = nib.load(path_BOLD).get_data()

        # Extract subject's behavioral data for the specified run
        path_behav = path_data + "behav/task001_run%s/behavdata.txt" % (run_id,)
        # Read in all but the first line, which is a just a header.
        raw = np.array([row.split() for row in list(open(path_behav))[1:]])
        kept_rows = raw[:, 4] != "0" if rm_nonresp else ...
        rare = raw[kept_rows].astype("float")
        # Volumes are captured every two seconds
        if time_correct: rare[:, 0] = rare[:, 0] // 2
        self.behav = np.array(rare[:, [0, 1, 2, 4, 5, 6]], dtype=int)

    def design_matrix(self, gain=True, loss=True, resp=False, resp_time = False,
                      euclidean_dist=True):
        """
        Creates the design matrix from the object's stored behavioral data.

        Parameters
        ----------
        gain : bool, optional
            True includes as a regressor parametric gains
        loss : bool, optional
            True includes as a regressor parametric losses
        resp : bool, optional
            True includes as a regressor the subject's response number
        resp_time: bool, optional
            True includes as a regressor the subject's response time
        euclidean_dist : bool, optional
            True includes the euclidean distance from the gain/loss combination
            to diagonal of the gain/loss matrix

        Return:
        ------
        design_matrix: 2-dimensional array
        resp_class: 1-dimensional array
            subject's response class (0 or 1)
        """
        # Determine which columns of behav to consider
        regressors = np.array([False, gain, loss, resp, False, resp_time])
        num_regressors = regressors.sum() + euclidean_dist
        design_matrix = np.ones((self.behav.shape[0], num_regressors + 1))
        design_matrix[:, 1:num_regressors] = self.behav[:, regressors]
        # the subject's response class (0 or 1)
        resp_class = self.behav[:,4]
        # Optional: Calculate the euclidean distance to the diagonal
        if euclidean_dist:
            gain, loss = self.behav[:, 1], self.behav[:, 2]
            gains = np.arange(10, 41, 2)
            # The euclidean distance of a point from the diagonal is the length
            # of the vector perpendicular to the diagonal and intersecting that
            # point. Take the gain/loss combination to be one vertex of an
            # isosceles right triangle. Then (`loss` - 5) gives the index of the
            # gain in `gains` of the point that lies both on the diagonal and on
            # the perpendicular vector defined above. Half the absolute value of
            # the difference between the observed `gain` and this calculated
            # gain (because `gains` increments by two) is the length of one leg
            # of our aforementioned triangle. We can then proceed to use this
            # leg to calculate the triangle's hypotenuse, which then gives the
            # penpendicular distance of the point to the diagonal.
            design_matrix[:, -1] = abs(gain - gains[loss - 5]) / np.sqrt(8)
        return design_matrix, resp_class

    def smooth(self, volume_number, sigma=1):
        """
        Returns a given volume of the BOLD data after application of a Gaussian
        filter with a standard deviation parameter of `sigma`.

        Parameters
        ----------
        volume_number : int
            Index of the desired volume of the BOLD data
        sigma : float, optional
            Standard deviation of the Gaussian kernel to be applied as a filter
        """
        input_slice = self.data[..., volume_number]
        smooth_data = gaussian_filter(input_slice, sigma)
        return smooth_data
