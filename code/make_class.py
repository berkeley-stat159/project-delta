from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

class run(object):
    """
    This class allows organization of the data by runs. Methods attached perform
    the indicated analyses of the data.
    """

    def __init__(self, sub_id, run_id, rm_nonresp=True):
        """
        Each object of this class created contains the fMRI data along with the
        corresponding behavioral data.

        Parameters
        ----------
        sub_id : str
            The unique key used to identify the subject (i.e., 001, ..., 016)
        run_id : str
            The unique key used to identify the run number (i.e, 001, ..., 003)
        rm_nonresp : bool, optional
            True removes trials that resulted in subject nonresponse
        """
        # Save the path to the directory containing the subject's data
        path_data = "../data/ds005/sub%s/" % (sub_id,)

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
        rare[:, 0] = rare[:, 0] // 2
        self.behav = np.array(rare[:, [0, 1, 2, 4, 5]], dtype=int)

    def design_matrix(self, gain=True, loss=True, resp=False, resp_bin=False
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
            True includes as a regressor the subject's response
        resp_bin : bool, optional
            True includes as a regressor the subject's response time
        euclidean_dist : bool, optional
            True includes the euclidean distance from the gain/loss combination
            to diagonal of the gain/loss matrix
        """
        # Determine which columns of behav to consider
        regressors = np.array([False, gain, loss, resp, resp_bin])
        num_regressors = regressors.sum() + euclidean_dist
        design_matrix = np.ones((self.behav.shape[0], num_regressors + 1))
        design_matrix[:, 1:num_regressors] = self.behav[:, regressors]
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
        return design_matrix
