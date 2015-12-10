"""
### Hypothesis Testing ###
1) T test for multiple linear regression

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.linalg as npl
from scipy.stats import t

def t_statistic(X, beta, response):
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
    t_stats : np.ndarray
        Array of shape (num_regressors, num_voxels) containing t-statistics for
        each estimated coefficient for each voxel
    p_vals : np.ndarray
        Array of shape (num_regressors, num_voxels) containing p-values that
        correspond to the given t-statistics
    """
    resids = response.T - X.dot(beta)
    RSS = np.sum(resids ** 2, axis=0)
    df = X.shape[0] - npl.matrix_rank(X)
    MSE = RSS / df
    MSE.shape = (MSE.shape[0], 1, 1)
    temp = np.diagonal(npl.pinv(X.T.dot(X)))
    std_err = np.sqrt(MSE * np.tile(temp, MSE.shape))
    std_err.shape = beta.shape
    zero = np.where(std_err == 0)
    std_err[zero] = 1
    t_value = beta / std_err
    p_vals = 2 * (1 - t.cdf(abs(t_value), df))
    return t_stats, p_vals
