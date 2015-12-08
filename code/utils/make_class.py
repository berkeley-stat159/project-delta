"""
This script contains code to create the ds005() class, which will automatically
complete all the grunt work required for a set of data before statistical
analyses can be performed on it. Future Python scripts can take advantage of the
ds005() class by including the command
    sys.path.append("code/utils")
    from make_class import *
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import sys
from scipy.ndimage.filters import gaussian_filter

sys.path.append("code/utils")
from hrf import *


class image(object):
    """
    """

    def __init__(self, path_image):
        """
        """
        #
        self.image = nib.load(path_image)
        self.data = self.image.get_data()
        self.affine = self.image.affine
        self.sigma = "Not ready yet"

        # Create specific sigma for filtered data, since we are using filtered data to plot
        # This self.sigma is voxel spefic               # This self.sigma is voxel spefic
        # For filtered data, the volumn per voxel (pixdim) is [2, 2, 2, 2]     +        # For filtered data, the volume per voxel (pixdim) is [2, 2, 2, 2]
        # 5mm FWHM = 2.355 sigma, keep the last dimension (time) 0              # 5mm FWHM = 2.355 sigma, keep the last dimension (time) 0
        #i_s = 5/2.355/2
        #j_s = 5/2.355/2
        #h_s = 5/2.355/2
        #self.sigma = [i_s, j_s, h_s, 0]
        #self.sigma_filtered = [i_s1, j_s1, h_s1, 0]

        #i_s1 = 5/2.355/2

        # For raw data, the volume per voxel (pixdim) is [3.125, 3.125, 4, 0]
        #i_s2 = 5/2.355/3.125
        #j_s2 = 5/2.355/3.125
        #h_s2 = 5/2.355/4
        #self.sigma_raw = [i_s2, j_s2, h_s2, 0]

    def smooth(self, sigma):
        """
        Returns a given volume of the BOLD data after application of a Gaussian
        filter with a standard deviation parameter of `sigma`
        
        Parameters
        ----------    
        sigma : 
            Standard deviation per voxel of the Gaussian kernel to be applied as a filter
            
        Return
        ------
        smooth_data : np.ndarray
           Array of shape self.data.shape
        """
        smooth_data = gaussian_filter(self.data, self.sigma)
        return smooth_data

    def convolution(self, regressor, step_size=2):
        """
        Computes the predicted convolved hemodynamic response function signals
        for a given regressor.

        Parameters
        ----------
        regressor : str
            Name of the regressor whose predicted neural time course and whose
            hemodynamic response function will be convolved
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
        corr : np.ndarray
            Array of shape (run.data.shape[:3],), where each value in 3-D space
            is the corresponding voxel's correlation coefficient of the BOLD
            signal with the specified regressor over time
        """
        time_course = self.time_course(regressor)
        n_voxels, n_volumes = np.prod(self.data.shape[:3]), self.data.shape[3]
        voxels = self.data.reshape(n_voxels, n_volumes)
        corr_1d = [np.corrcoef(voxel, time_course)[0, 1] for voxel in voxels]
        corr = np.reshape(corr_1d, self.data.shape[:3])
        return corr


class ds005(object):
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
            Unique key used to identify the subject (i.e., 001, ..., 016)
        run_id : str
            Unique key used to identify the run number (i.e, 001, ..., 003)
        rm_nonresp : bool, optional
            True removes trials that resulted in subject nonresponse
        """
        # Save the path to the directory containing the subject's data
        path_data = "data/ds005/sub%s/" % sub_id
        path_run = "task001_run%s" % run_id

        # Extract subject's behavioral data for the specified run
        path_behav = path_data + "behav/" + path_run + "/behavdata.txt"
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

        # Load filtered and raw fMRI images
        self.filtered = image(path_data + "model/model001/" + path_run +
                              ".feat/filtered_func_data_mni.nii.gz")
        self.raw = image(path_data + "BOLD/" + path_run + "/bold.nii.gz")

        # Extract subject's task condition data
        path_cond = path_data + "model/model001/onsets/" + path_run
        conditions = ()
        for condition in range(2, 5):
            raw_matrix = list(open(path_cond + "/cond00%s.txt" % condition))
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
        design_matrix : np.ndarray
            Design matrix from subjects' behavioral data with a column for each
            desired regressor and a row for each desired trial
        """
        # Determine which columns of behav to consider as regressors
        columns = [False, gain, loss, euclidean_dist, False, False, resp_time]
        n_regressors = columns.count(True) + 1
        design_matrix = np.ones((self.behav.shape[0], n_regressors))
        design_matrix[:, 1:n_regressors] = self.behav[:, np.array(columns)]
        return design_matrix
