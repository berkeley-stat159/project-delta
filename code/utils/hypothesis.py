"""
This script contains code that allows quick and easy assessment of the
statistical significance of the proposed explanatory variables of a regression
or classification model used in our study. Future Python scripts can take
advantage of this module by including the command
    sys.path.append("code/utils")
    from hypothesis import *
"""
from __future__ import division, print_function, absolute_import
from scipy.stats import norm, t
import numpy as np
import numpy.linalg as npl

def ttest(X, beta, response):
    """
    Performs a t-test on the results of a multiple linear regression.

    Parameters
    ----------
    X : np.ndarray
        Return from ds005.design_matrix() method
    beta : np.ndarray
        Array of shape (num_regressors, num_voxels) containing the estimated
        values of beta
    response : np.ndarray
        2-D array of BOLD data

    Return
    ------
    t_stat : np.ndarray
        Array of shape (num_regressors, num_voxels) containing t-statistics for
        each estimated coefficient for each voxel
    p_value : np.ndarray
        Array of shape (num_regressors, num_voxels) containing p-values that
        correspond to the given t-statistics
    """
    resids = response.T - X.dot(beta)
    RSS = np.sum(resids ** 2, axis=0)
    df = X.shape[0] - npl.matrix_rank(X)
    MSE = RSS / df
    MSE.shape = (MSE.shape[0], 1, 1)
    temp = np.diagonal(npl.pinv(X.T.dot(X)))
    st_err = np.sqrt(MSE * np.tile(temp, MSE.shape))
    st_err.shape = beta.shape
    zeros = np.where(st_err == 0)
    st_err[zeros] = 1
    t_stat = beta / st_err
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    return t_stat, p_value

def waldtest(design_matrix, beta_hat, prob_estimates):
    """
    Performs a Wald test to assess the statistical significance of each of a
    given number of regressors.

    Parameters
    ----------
    design_matrix : np.ndarray
        Output returned by ds005.design_matrix() method
    beta_hat : np.ndarray
        Array of shape (num_regressors + 1,) of coefficients for regressors
        estimated by LogisticRegression() class
    prob_estimates : np.ndarray
        Array of shape (num_trials, num_trials) containing probability estimates
        for prediction by logistic regression model

    Return
    ------
    p_value : np.ndarray
        Array of shape (num_regressors + 1,) containing estimates of the
        statistical significance of each regressor given
    """
    assert design_matrix.shape[0] == prob_estimates.shape[0], "shape mismatch"
    var = np.diag(design_matrix.shape[0] * np.product(prob_estimates, axis=1))
    inv_sym = npl.inv(design_matrix.T.dot(var.dot(design_matrix)))
    st_err = np.sqrt(np.diagonal(abs(inv_sym)))
    z_stat = beta_hat / st_err
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return p_value
