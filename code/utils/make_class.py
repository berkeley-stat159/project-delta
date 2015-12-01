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

    def __init__(self, sub_id, run_id, rm_nonresp=True, filtered_data=False):
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
        filtered_data : bool, optional
            True uses the filtered BOLD data; else uses the raw BOLD data
        """
        if path:
            path_data = "../../data/ds005/sub%s/" % sub_id
        else:
            # Save the path to the directory containing the subject's data
            path_data = "data/ds005/sub%s/" % sub_id

        # Extract subject's behavioral data for the specified run
        path_behav = path_data + "behav/task001_run%s/behavdata.txt" % run_id
        # Read in all but the first line, which is a just a header.
        raw = np.array([row.split() for row in list(open(path_behav))[1:]])
        kept_rows = raw[:, 4] != "0" if rm_nonresp else np.arange(raw.shape[0])
        rare = raw[kept_rows].astype("float")
        # Calculate the euclidean distance to the diagonal
        gain, loss = rare[:, 1], rare[:, 2].astype(int)
        gains = np.arange(10, 41, 2)
        # The euclidean distance of a point from the diagonal is the length of
        # the vector intersecting that point and orthogonal to the diagonal.
        # Take the gain/loss combination to be one vertex of an isosceles right
        # triangle. Then (`loss` - 5) gives the index of the gain in `gains` of
        # the point that lies both on the diagonal and on the orthogonal vector
        # defined above. Half the absolute value of the difference between the
        # observed `gain` and this calculated gain (because `gains` increments
        # by two) is the length of one leg of our triangle. We can then proceed
        # to use this leg to calculate the triangle's hypotenuse, which then
        # gives the perpendicular distance of the point to the diagonal.
        rare[:, 3] = abs(gain - gains[loss - 5]) / np.sqrt(8)
        self.behav = rare

        # Extract subject's BOLD signal data for the specified run
        if filtered_data:
            path_BOLD = (path_data + "model/model001/task001_run%s.feat/" +
                         "filtered_func_data_mni.nii.gz") % run_id
        else:
            path_BOLD = path_data + "BOLD/task001_run%s/bold.nii.gz" % run_id
        self.affine = nib.load(path_BOLD).affine
        self.data = nib.load(path_BOLD).get_data()

        # Extract subject's task condition data
        path_cond = path_data + "model/model001/onsets/task001_run%s/" % run_id
        conditions = ()
        for condition in range(2, 5):
            raw_matrix = list(open(path_cond + "cond00%s.txt" % condition))
            cond = np.array([row.split() for row in raw_matrix]).astype("float")
            conditions += (cond,)
        self.cond_gain, self.cond_loss, self.cond_dist_from_indiff = conditions

    def design_matrix(self, gain=True, loss=True, euclidean_dist=True,
                      resp_time=False):
        """
        Creates the design matrix from the object's stored behavioral data.

        Parameters
        ----------
        gain : bool, optional
            True includes as a regressor parametric gains
        loss : bool, optional
            True includes as a regressor parametric losses
        euclidean_dist : bool, optional
            True includes the euclidean distance from the gain/loss combination
            to diagonal of the gain/loss matrix
        resp_time : bool, optional
            True includes as a regressor subject response time

        Return
        ------
        Design matrix from subjects' behavioral data with a column for each
        desired regressor and a row for each desired trial
        """
        # Determine which columns of behav to consider as regressors
        columns = [False, gain, loss, euclidean_dist, False, False, resp_time]
        n_regressors = columns.count(True) + 1
        design_matrix = np.ones((self.behav.shape[0], n_regressors))
        design_matrix[:, 1:n_regressors] = self.behav[:, np.array(columns)]
        return design_matrix

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

        Return
        ------
        Numpy array of shape run.data.shape[:3], where each value in three-
        dimensional space is the corresponding voxel's BOLD signal strength
        after smoothing with a Gaussian filter.
        """
        input_slice = self.data[..., volume_number]
        smooth_data = gaussian_filter(input_slice, sigma)
        return smooth_data

    def time_course(self, regressor, step_size=2):
        """
        Generates predictions for the neural time course, with respect to a
        regressor.

        Parameters
        ----------
        regressor : str
            Name of regressor whose amplitudes will be used to generate the
            time course: select from "gain", "loss", "dist_from_indiff"
        step_size : float, optional
            Size of temporal steps (in seconds) at which to generate predictions
        trial_length : float, optional
            Time alloted to subject to complete each trial of the task

        Return
        ------
        One-dimensional numpy array, containing 0s for time between trials and
        values defined by the specified regressor for time during trials.
        """
        condition = {"gain": self.cond_gain, "loss": self.cond_loss,
                     "dist_from_indiff": self.cond_dist_from_indiff}[regressor]
        onsets = condition[:, 0] / step_size
        periods, amplitudes = condition[:, 1] / step_size, condition[:, 2]
        # Time resolution of the BOLD data is two seconds
        time_course = np.zeros(int(2 * self.data.shape[3] / step_size))
        for onset, period, amplitude in list(zip(onsets, periods, amplitudes)):
            onset, period = int(np.floor(onset)), int(np.ceil(period))
            time_course[onset:(onset + period)] = amplitude
        return time_course

    def correlation(self, regressor):
        """
        Calculates the correlation coefficient of the BOLD signal with a single
        regressor for each voxel across time.

        Parameters
        ----------
        regressor : str
            Name of regressor whose correlation with the BOLD data is of
            interest: select from "gain", "loss", "dist_from_indiff"

        Return
        ------
        Numpy array of shape run.data.shape[:3], where each value in three-
        dimensional space is the corresponding voxel's correlation coefficient
        of the BOLD signal with the specified regressor over time.
        """
        time_course = self.time_course(regressor)
        n_voxels, n_volumes = np.prod(self.data.shape[:3]), self.data.shape[3]
        voxels = self.data.reshape(n_voxels, n_volumes)
        corr_1d = [np.corrcoef(voxel, time_course)[0, 1] for voxel in voxels]
        corr = np.reshape(corr_1d, self.data.shape[:3])
        return corr
