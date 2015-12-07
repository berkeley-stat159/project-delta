"""
Class analysis
Inherit from class run
This class consists analysis tools for fMRI data

Call "make analysis" in the main directory of the project folder
The folder "results" will create in 
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from plot_tool import *
import sys
sys.path.append("code/utils")
from make_class import *
from convolution import *
from diagnostics import *
from hypothesis import *

class analysis(run):
    """
    Inherit from run class and perfrom analysis on the specified run

    """
    TR = 2
    n_trs = 240
    tr_divs = 100
    
    def behav_analysis(self, euclidean_dist=True):
        """
        perform logistic regression on behavior data

        Parameter:
        ----------
        euclidean_dist: boolean (default: True)
            True if including euclidean distance as a regressor in logistic regression

        Return:
        -------
        behav_beta: array (# of regressors+1, )
            regression coeffiecients for intercept, gain, loss, euclidean_dist(optional)
        behav_lam: float
            loss aversion
        misclass_rate: float
            misclassfication rate of logistic model
        behav_p_value:
            corresponding p-value for t test for each coeffiecient
            which the null hypothesis is the coeffiecient is zero
        """
        # Step 1: Construct design matrix
        X = self.design_matrix(gain=True, loss=True, euclidean_dist=euclidean_dist)
        Y = self.behav[:,5]
        
        # Step 2: Create logistic model
        log_mod = LogisticRegression()

        # Step 3: fit training data
        log_mod.fit(X,Y)

        # Step 4: coeffiecients
        beta_hat = log_mod.coef_
        behav_beta = beta_hat.ravel()
        behav_lam = -behav_beta[2]/behav_beta[1]

        # Step 5: Training accuracy
        N = X.shape[0]
        prob = log_mod.predict_proba(X)
        pred = np.zeros(N)
        pred[prob[:,1]>0.5] = 1
        misclass_rate = np.sum(pred!=Y)/N

        # Step 6: Wald test
        var = np.diag(N*np.product(prob,axis=1))
        S = npl.inv(X.T.dot(var.dot(X)))
        SE = np.sqrt(np.diagonal(S))
        z = behav_beta/SE
        behav_p_value = (1-norm.cdf(abs(z)))*2
        return behav_beta, behav_lam, misclass_rate, behav_p_value

    def bold_predict(self):
        """
        Predict BOLD signal by convolution with hrf
        """
        self.neural_gain = self.time_course("gain", step_size=0.02) # where 0.02 = TR/tr_divs
        self.neural_loss = self.time_course("loss", step_size=0.02)
        self.neural_dist = self.time_course("dist_from_indiff", step_size=0.02)

        hrf_times = np.arange(0,30,self.TR/self.tr_divs)
        hrf_at_hr = hrf(hrf_times)

        self.hemo_pred_gain = np.convolve(self.neural_gain, hrf_at_hr)[:len(self.neural_gain)]
        self.hemo_pred_loss = np.convolve(self.neural_loss, hrf_at_hr)[:len(self.neural_loss)]
        self.hemo_pred_dist = np.convolve(self.neural_dist, hrf_at_hr)[:len(self.neural_dist)]
        
    def bold_figure(self, save=False, path=None):
        """
        Show BOLD prediction figures

        Parameters:
        ----------
        save: bool
            True if want to save the figure as an output png file
        """
        all_tr_times = np.arange(0, self.n_trs, 1/self.tr_divs)*self.TR

        plt.figure(figsize=(15, 10), dpi=100)
        plt.subplot(231)
        plt.plot(all_tr_times, self.hemo_pred_gain)
        plt.xlabel("Time")
        plt.ylabel("Hemodynamic Response")
        plt.title("Condition 1: Gain")

        plt.subplot(232)
        plt.plot(all_tr_times, self.hemo_pred_loss)
        plt.xlabel("Time")
        plt.ylabel("Hemodynamic Response")
        plt.title("Condition 2: Loss")

        plt.subplot(233)
        plt.plot(all_tr_times, self.hemo_pred_dist)
        plt.xlabel("Time")
        plt.ylabel("Hemodynamic Response")
        plt.title("Condition 3: Distance from Indifference")

        plt.subplot(234)
        plt.plot(all_tr_times, self.neural_gain)
        plt.xlabel("Time")
        plt.ylabel("Neural Prediction")
        plt.title("Condition 1: Gain")
        
        plt.subplot(235)
        plt.plot(all_tr_times, self.neural_loss)
        plt.xlabel("Time")
        plt.ylabel("Neural Prediction")
        plt.title("Condition 2: Loss")
        
        plt.subplot(236)
        plt.plot(all_tr_times, self.neural_dist)
        plt.xlabel("Time")
        plt.ylabel("Neural Prediction")
        plt.title("Condition 3: Distance from Indifference")

        if save:
            plt.savefig(path+'convolution3cond.png',dpi=500)
            plt.close()

    def outlier_detection(self):
        """
        Detect extend difference-based outliers

        Return:
        ------
        edo_index: 1-dimensional array
            Extend difference-based outlier indices
        """
        #Return root mean square of differences between sequential volumes
        rmsd = vol_rms_diff(self.data)
        #Return indices of outliers identified by interquartile range
        rmsd_outlier_id = iqr_outliers(rmsd)[0]
        #Extend difference-based outlier indices
        edo_index = extend_diff_outliers(rmsd_outlier_id)
        return edo_index

    def linear_analysis(self, rm_outlier=False):
        """
        Perform linear analysis on BOLD fmri image data

        Parameter:
        ---------
        rm_outlier: bool (default: False)
            True if removing outiliers before performing linear regression

        Return:
        ------
        RSS: array (236, )
            Residual Sum of Squares
        MRSS: float
            Mean Residual Sum of Squares
        """
        self.data = self.data[...,4:]
        index = np.arange(0,self.n_trs*self.tr_divs, self.tr_divs)
        convolved1 = self.hemo_pred_gain[index][4:] 
        convolved2 = self.hemo_pred_loss[index][4:] 
        convolved3 = self.hemo_pred_dist[index][4:] 

        N = len(convolved1)
        self.X = np.ones((N, 4)) #make a design matrix for linear regression

        self.X[:, 0] = convolved1 # gain
        self.X[:, 1] = convolved2 # loss
        self.X[:, 2] = convolved3 # euclidean dist

        self.data2d = np.reshape(self.data, (np.prod(self.data.shape[:-1]), -1))
        # remove outliers if rm_outlier = True
        if rm_outlier:
            edo_index = self.outlier_detection()
            X = np.delete(self.X, edo_index, 0)
            self.data2d = np.delete(self.data2d, edo_index, 1)
        data2d_trans = self.data2d.T
        Xp = npl.pinv(X)
        self.neural_beta = Xp.dot(data2d_trans)

        # calculate MRSS
        MRSS = np.ones(self.data2d.shape[0])
        res = data2d_trans - self.X.dot(self.neural_beta)
        RSS = np.sum(res**2, axis=0)
        df = self.X.shape[0] - npl.matrix_rank(self.X)
        MRSS = RSS / df
        return RSS, MRSS

    def locate_activated_voxel(self, regressor):
        """ Find the activated voxels

        Parameters:
        ----------
        regressor: str
            "gain", "loss", or "dist_from_indiff"
        Return:
        ------
        active_voxel: 2-dimensional array
            position of voxel in BOLD data
        """
        reg = {"gain": 0, "loss": 1, "dist_from_indiff": 2}[regressor]
        t, p = t_statistic(self.X, self.neural_beta, self.data2d)

        p = p[reg,:]
        p.shape = self.data.shape[:-1]
        beta = self.neural_beta[reg,:]
        beta.shape = self.data.shape[:-1]
        thres = np.mean(beta)
        active_voxel = np.transpose(((beta > thres) & (p < 0.05)).nonzero())
        return active_voxel

    def convert_to_coordinate(self, active_voxel):
        """ Find the actual coordinate in brain corresponding to each voxel position

        Parameters:
        ----------
        active_voxel: 2-D array
            postition of activated voxel, returned by locate_activated_voxel

        Return:
        ------
        coordinate: 2-D array, same shape as active_voxel
            actual coordinate in brain templete

        """
        vox_to_mm = self.affine
        coordinate = nib.affines.apply_affine(vox_to_mm, active_voxel)
        return coordinate


if __name__ == "__main__":
    import os
    sub_id = ["001","002","003","004","005","006","007","008","009","010","011","012","013","014","015","016"]
    run_id = ["001","002","003"]
    pair_id = list(zip([i for _ in range(3) for i in sub_id], [i for i in run_id for _ in range(16)]))

    path = "results"
    if not os.path.exists(path):
        os.makedirs(path)
        for i in sub_id:
            os.makedirs(path+"/sub"+i)
            for j in run_id:
                os.makedirs(path+"/sub"+i+"/run"+j)

    # the set of lambdas (with euclidean dist) for each run and subject (16x3 total)
    lambda_set1 = []
    # the set of lambdas (without euclidean dist) for each run and subject (16x3 total)
    lambda_set2 = []

    for p in pair_id:
        # s: sub, r: run
        s, r = p
        # saving path
        save_path = "results/sub%s/run%s/" % (s, r)
        # create an object called "subject"
        subject = analysis(s, r)
        # Logistic regression
        behav_beta1, behav_lam1, misclass_rate1, behav_p_value1 = subject.behav_analysis(euclidean_dist=True)
        behav_beta2, behav_lam2, misclass_rate2, behav_p_value2 = subject.behav_analysis(euclidean_dist=False)
        # storing lambda for each run
        lambda_set1.append(behav_lam1)
        lambda_set2.append(behav_lam2)
        # write logistic model results to txt files
        with open(save_path+"logistic_results.txt", "w") as f:
            f.write("With euclidean distance from Indifference\nCoefficients: %s\nLoss aversion (lambda): %s \nMisclassification rate: %s \nP value: %s \n\nWithout euclidean distance from Indifference\nCoefficients: %s \nLoss aversion (lambda): %s \nMisclassification rate: %s \nP value: %s \n" \
                % (behav_beta1, behav_lam1, misclass_rate1, behav_p_value1,behav_beta2, behav_lam2, misclass_rate2, behav_p_value2))
        # neural stimulus convolve with hrf
        subject.bold_predict()
        # plot neural prediction and hemodynamic response prediction over time
        subject.bold_figure(save=True, path=save_path)
        # Linear analysis
        



