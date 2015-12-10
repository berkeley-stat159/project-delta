"""
Purpose:
This script fits a logistic regression model to predict the subject's response,
for each run, using three regressors: parametric gain, parametric loss, and the
euclidean distance of the gain/loss combination from the diagonal of the
gain/loss matrix.

It should return ###############################################################
"""
from __future__ import absolute_import, division, print_function
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import numpy as np
import numpy.linalg as npl
import sys

sys.path.append("code/utils")
from make_class import *


# Create a collection of all subject IDs and all run IDs
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
IDs = list(zip([subject_ID for _ in range(3) for subject_ID in subject_IDs],
              [run_ID for _ in range(16) for run_ID in run_IDs]))


# Perform the procedure outlined here on each run of each subject:
for ID in IDs:
    subject, run = ID


    # Import the necessary data:
    # - X will be our design matrix. By default, the .design_matrix() method
    #   produces a numpy array with exactly four columns and as many rows as
    #   trials that resulted in some response. The first columns contains all
    #   ones, the next gain in dollars, the third loss in dollars, and the last
    #   euclidean distance in units-distance with respect to the gamble matrix.
    # - Y will be a one-dimensional numpy array containing the subject's binary
    #   responses, 0 representing decline and 1 representing acceptance.
    obj = ds005(subject, run)
    X = obj.design_matrix()
    responses = obj.behav[:, 5]


    # Create an instance of LogisticRegression() and fit the data
    log_model = LogisticRegression().fit(X, responses)


    # Use the class to estimate our four coefficients
    beta_hat = log_model.coef_.ravel()
    print("The estimated coefficients are {}".format(beta_hat))
    lamda = -beta_hat[2] / beta_hat[1]
    print("The behavioral loss aversion score (lambda) is {}".format(lamda))


    # Next, we'll check our model's accuracy
    num_rows = X.shape[0]
    probability_estimates = log_mod.predict_proba(X)
    predictions = np.zeros(num_rows)
    pred[probability_estimates[:, 1] > 0.5] = 1
    misclassification_rate = np.sum(predictions != responses) / num_rows
    print("The misclassification rate is {}".format(misclassification_rate))


    # Lastly, we perform a Wald test to access the statistical significance of
    # each of the three regressors
    variances = np.diag(num_rows * np.product(probability_estimates, axis=1))
    std_devs = npl.inv(X.T.dot(variances.dot(X)))
    std_errs = np.sqrt(np.diagonal(std_devs))
    z_stats = beta_hat / std_errs
    p_values = 2 * (1 - norm.cdf(abs(z_stats)))
    print("The p values for each coeffiecient are {}".format(p_values))
    print("=" * 80 + "\n")


    # Now, we'll perform one more analysis, without euclidean distance as a
    # regressor. It's a good idea to do this as euclidean distance is partially
    # dependent on both the parametric gain and the parametric loss, so its
    # inclusion may in fact underestimate the effect of the other two.
    X2 = obj.design_matrix(euclidean_dist=False)


    # Create another fitted instance of LogisticRegression()
    log_model2 = LogisticRegression().fit(X2, responses)


    # Estimate our three coefficients
    beta_hat2 = log_model2.coef_.ravel()
    print("The esitmated coeffiecients (without euclidean distance) are" +
          " {}".format(beta_hat2))
    lamda2 = -beta_hat2[2] / beta_hat2[1]
    print("The Behavioral loss aversion (lambda) is {}".format(lamda2))


    # Calculate the new model's rate of misclassification
    probability_estimates2 = log_mod2.predict_proba(X2)
    predictions2 = np.zeros(num_rows)
    predictions2[probability_estimates2[:, 1] > 0.5] = 1
    print("The misclassification rate is {}".format(misclassification_rate2))


    # Another Wald test to access the statistical significance of our two
    # regressors, without euclidean distance
    variances2 = np.diag(num_rows * np.product(probability2, axis=1))
    std_devs2 = npl.inv(X2.T.dot(variances2.dot(X2)))
    std_errs2 = np.sqrt(np.diagonal(std_devs2))
    z_stats2 = beta_hat2 / std_errs2
    p_values2 = 2 * (1 - norm.cdf(abs(z_stats2)))
    print("The p values for each coeffiecient are {}".format(p_values2))
