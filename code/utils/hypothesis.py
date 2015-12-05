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
    perform t-test on multiple linear regression

    Parameters:
    ----------
    X: 2D array (number of TRs, number of regressors)
        design matrix
    beta: 2D array (number of regressors, number of voxels)
        estimated beta values
    response: 2D array (number of voxels, number of TRs)
        2D BOLD matrix

    Return:
    ------
    t_value: 2D array (number of regressors, number of voxels)
        t values for each voxel corresponding to each regressor
    p: 2D array (number of regressors, number of voxels)
        significance level
    """
    res = response.T - X.dot(beta)
    RSS = np.sum(res**2, axis=0)
    df = X.shape[0] - npl.matrix_rank(X)
    MSE = RSS / df
    MSE.shape = (MSE.shape[0],1,1)
    temp = np.diagonal(npl.pinv(X.T.dot(X)))
    SE = np.sqrt(MSE * np.tile(temp, (MSE.shape[0],1,1)))
    SE.shape = beta.shape
    zero = np.where(SE==0)
    SE[zero] = 1
    t_value = beta / SE
    p = (1-t.cdf(abs(t_value),df))*2
    return t_value, p


