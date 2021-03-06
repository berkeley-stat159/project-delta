"""
This script contains code to create the img() and ds005() classes, which will
automatically complete all the grunt work required for a set of data before
statistical analyses can be performed on it. Future Python scripts can take
advantage of the img() and ds005() classes by including the command
    sys.path.append("code/utils")
    from make_class import *
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os, sys
from scipy.ndimage.filters import gaussian_filter

sys.path.append("code/utils")
from hrf import *


class img(object):
    """
    This class organizes each file containing fMRI data and provides a quick way
    to extract crucial information necessary for later statistical analyses.
    """

    def __init__(self, file_path):
        """
        Each object of this class created will contain the fMRI data that comes
        from a single file. While keeping the original image, it also saves
        critical attributes attached to the image for easy access. This class is
        meant to be used exclusively within the ds005() class.

        Parameters
        ----------
        file_path : str
            Path leading from the main project directory to the file containing
            the fMRI BOLD signal data of interest
        """
        # Load the fMRI image saved to the specified file
        assert os.path.isfile(file_path), "nonexistent file for subject/run"
        self.img = nib.load(file_path)
        
        # Extract the BOLD data enclosed within the image
        self.data = self.img.get_data()
        
        # Extract the affine of the fMRI image
        self.affine = self.img.affine

        # Extract the voxel to mm conversion rate from the image affine
        mm_per_voxel = abs(self.affine.diagonal()[:3])
        self.voxels_per_mm = np.append(np.reciprocal(mm_per_voxel), 0)

    def smooth(self, fwhm=5):
        """
        Returns a given volume of the BOLD data after application of a Gaussian
        filter with a standard deviation parameter of `sigma`

        Parameters
        ----------
        fwhm : float or np.ndarray(..., dtype=float), optional
            Millimeter measurement of the full-width-at-half-maximum of the
            Gaussian distribution whose kernel will be used in smoothing. If
            np.ndarray(), shape must be (4,)

        Return
        ------
        smooth_data : np.ndarray
           Array of shape self.data.shape
        """
        if type(fwhm) == np.ndarray:
            assert fwhm.shape == (4,), "invalid shape in fwhm"
            assert fwhm.dtype in ["float_", "int_"], "invalid dtype in fwhm"
        else:
            assert type(fwhm) in [float, int], "invalid type in fwhm"
        sigma_in_voxels = fwhm / np.sqrt(8 * np.log(2)) * self.voxels_per_mm
        smooth_data = gaussian_filter(self.data, sigma_in_voxels)
        return smooth_data


class ds005(object):
    """
    This class allows organization of the data by runs. In addition to the
    behavioral data, it also contains as subobjects the raw and filtered data.
    """

    def __init__(self, sub_id, run_id, rm_nonresp=True):
        """
        Each object of this class created contains both sets of fMRI data along
        with the corresponding behavioral data.
        
        Parameters
        ----------
        sub_id : str
            Unique key used to identify the subject (i.e., 001, ..., 016)
        run_id : str
            Unique key used to identify the run number (i.e, 001, ..., 003)
        rm_nonresp : bool, optional
            True removes trials that resulted in subject nonresponse
        """
        # Save parts of the paths to the directories containing the data
        path_sub = "data/ds005/sub%s/" % sub_id
        path_run = "task001_run%s" % run_id

        # Extract subject's behavioral data for the specified run
        path_behav = path_sub + "behav/" + path_run + "/behavdata.txt"
        # Read in all but the first line, which is a just a header.
        raw = np.array([row.split() for row in list(open(path_behav))[1:]])
        kept_rows = raw[:, 4] != "0" if rm_nonresp else np.arange(raw.shape[0])
        rare = raw[kept_rows].astype("float")
        # Calculate the distance to indifference--defined to be the euclidean
        # distance from the gain/loss combination to the diagonal of the
        # gain/loss matrix
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

        # Extract subject's task condition data
        path_cond = path_sub + "model/model001/onsets/" + path_run
        conditions = ()
        for condition in range(2, 5):
            raw_matrix = list(open(path_cond + "/cond00%s.txt" % condition))
            cond = np.array([row.split() for row in raw_matrix]).astype("float")
            conditions += (cond,)
        self.cond_gain, self.cond_loss, self.cond_dist2indiff = conditions

        # Load raw and filtered fMRI images
        self.raw = img(path_sub + "BOLD/" + path_run + "/bold.nii.gz")
        self.filtered = img(path_sub + "model/model001/" + path_run +
                            ".feat/filtered_func_data_mni.nii.gz")

    def design_matrix(self, gain=True, loss=True, dist2indiff=True,
                      resp_time=False):
        """
        Creates the design matrix from the object's stored behavioral data.
        
        Parameters
        ----------
        gain : bool, optional
            True includes as a regressor parametric gains
        loss : bool, optional
            True includes as a regressor parametric losses
        dist2indiff : bool, optional
            True includes the regressor distance to indifference
        resp_time : bool, optional
            True includes as a regressor subject response time
            
        Return
        ------
        design_matrix : np.ndarray
            Design matrix from subjects' behavioral data with a column for each
            desired regressor and a row for each desired trial
        """
        # Determine which columns of behav to consider as regressors
        columns = [False, gain, loss, dist2indiff, False, False, resp_time]
        n_regressors = columns.count(True) + 1
        design_matrix = np.ones((self.behav.shape[0], n_regressors))
        design_matrix[:, 1:n_regressors] = self.behav[:, np.array(columns)]
        return design_matrix
    
    def time_course(self, regressor, step_size=2):
        """
        Generates predictions for the neural time course, with respect to a
        regressor.
        
        Parameters
        ----------
        regressor : str
            Name of regressor whose amplitudes will be used to generate the
            time course: select from "gain", "loss", "dist2indiff"
        step_size : float, optional
            Size of temporal steps (in seconds) at which to generate predictions
            
        Return
        ------
        time_course : np.ndarray
            1-D numpy array, containing 0s for time between trials and values
            defined by the specified regressor for time during trials
        """
        assert regressor in ["gain", "loss", "dist2indiff"], "invalid regressor"
        condition = {"gain": self.cond_gain, "loss": self.cond_loss,
                     "dist2indiff": self.cond_dist2indiff}[regressor]
        onsets = condition[:, 0] / step_size
        periods, amplitudes = condition[:, 1] / step_size, condition[:, 2]
        # The default time resolution in this study was two seconds
        time_course = np.zeros(int(2 * self.raw.data.shape[3] / step_size))
        for onset, period, amplitude in list(zip(onsets, periods, amplitudes)):
            onset, period = int(np.floor(onset)), int(np.ceil(period))
            time_course[onset:(onset + period)] = amplitude
        return time_course

    def convolution(self, regressor, step_size=2):
        """
        Computes the predicted convolved hemodynamic response function signals
        for a given regressor.

        Parameters
        ----------
        regressor : str
            Name of the regressor whose predicted neural time course and whose
            hemodynamic response function will be convolved: select from "gain",
            "loss", "dist2indiff"
        step_size : float
            Size of temporal steps (in seconds) at which to generate signals

        Return
        ------
        convolution : np.ndarray
            Array containing the predicted hemodynamic response function values
            for the given regressor
        """
        time_course = self.time_course(regressor, step_size)
        # Hemodynamic responses typically last 30 seconds
        hr_func = hrf(np.arange(0, 30, step_size))
        convolution = np.convolve(time_course, hr_func)[:len(time_course)]
        return convolution
