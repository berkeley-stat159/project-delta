"""
Purpose:
This script fits a logistic regression model to predict the subject's response,
for each run, using three regressors: parametric gain, parametric loss, and the
the distance from indifference.

It should produce two plaintext files per run, which contain the results of
fitting a logistic model. In addition, it should produce two more collective
plaintext files, which contain the values of Lambda, one with the distance to
indifference as a regressor and one without.
"""
from __future__ import absolute_import, division, print_function
from sklearn.linear_model import LogisticRegression
import numpy as np
import numpy.linalg as npl
import sys

sys.path.append("code/utils")
from hypothesis import *
from make_class import *


# Create plaintext files in which to store Lambdas
with open("results/lambda_dist2indiff.txt", "w") as outfile:
        outfile.write("run\tsubject\tlambda\n")
with open("results/lambda_no_dist2indiff.txt", "w") as outfile:
        outfile.write("run\tsubject\tlambda\n")


# Create a collection of all subject IDs and all run IDs
run_IDs = [str(i).zfill(3) for i in range(1, 4)]
subject_IDs = [str(i).zfill(3) for i in range(1, 17)]
IDs = list(zip([run_ID for _ in range(16) for run_ID in run_IDs],
               [subject_ID for _ in range(3) for subject_ID in subject_IDs]))
IDs.sort()

# We perform the procedure outlined in this script for each run of each subject:
for ID in IDs:
    run, subject = ID


    # Create a directory to store the script's output
    path_result = "results/run%s/logistic/sub%s/" % ID
    try:
        os.makedirs(path_result)
    except OSError:
        if not os.path.isdir(path_result):
            raise


    # Import the necessary data:
    # - X will be our design matrix. By default, the .design_matrix() method
    #   produces a numpy array with exactly four columns and as many rows as
    #   trials that resulted in some response. The first columns contains all
    #   ones, the next gain in dollars, the third loss in dollars, and the last
    #   distance to indifference in units-distance with respect to the gamble
    #   matrix.
    # - Y will be a one-dimensional numpy array containing the subject's binary
    #   responses, 0 representing decline and 1 representing acceptance.
    obj = ds005(subject, run)
    X = obj.design_matrix()
    Y = obj.behav[:, 5]


    # Create an instance of LogisticRegression() and fit the data
    log_model = LogisticRegression().fit(X, Y)


    # Use the class to estimate our four coefficients
    regr_coef = log_model.coef_.ravel()
    regr_coef_tuple = tuple(value for value in regr_coef)
    with open(path_result + "results.txt", "w") as outfile:
        newline = ("dist2indiff_included:\n\n" +
                   "regr_coef:\n" +
                   ("\tintercept:\t%s\n" % regr_coef_tuple[0]) +
                   ("\tparam_gain:\t%s\n" % regr_coef_tuple[1]) +
                   ("\tparam_loss:\t%s\n" % regr_coef_tuple[2]) +
                   ("\teuclid_dist:\t%s\n\n" % regr_coef_tuple[3]))
        outfile.write(newline)


    # Compute Lambda and save its value to `lambda.txt`
    Lambda = -regr_coef[2] / regr_coef[1]
    with open("results/lambda_dist2indiff.txt", "a") as outfile:
        newline = (run + "\t" + subject + "\t" + str(Lambda) + "\n")
        outfile.write(newline)


    # Next, we'll check our model's accuracy
    num_rows = X.shape[0]
    probability_estimates = log_model.predict_proba(X)
    predictions = np.zeros(num_rows)
    predictions[probability_estimates[:, 1] > 0.5] = 1
    misclassification_rate = np.sum(predictions != Y) / num_rows
    with open(path_result + "results.txt", "a") as outfile:
        newline = ("misclass_rate:\t%s\n\n" % misclassification_rate)
        outfile.write(newline)


    # Lastly, we perform a Wald test to assess the statistical significance of
    # each of the three regressors
    p_values = waldtest(X, regr_coef, probability_estimates)
    p_values_tuple = tuple(value for value in p_values)
    with open(path_result + "results.txt", "a") as outfile:
        newline = ("stat_signif:\n" +
                   ("\tintercept:\t%s\n" % p_values_tuple[0]) +
                   ("\tparam_gain:\t%s\n" % p_values_tuple[1]) +
                   ("\tparam_loss:\t%s\n" % p_values_tuple[2]) +
                   ("\teuclid_dist:\t%s\n\n\n" % p_values_tuple[3]))
        outfile.write(newline)


    # Now, we'll perform one more analysis, without the distance to indifference
    # as a regressor. It's a worthwhile idea to do this as the distance to
    # indifference is partially dependent on both the parametric gain and the
    # parametric loss, so its inclusion may in fact underestimate the effect of
    # the other two.
    X2 = obj.design_matrix(dist2indiff=False)


    # Create another fitted instance of LogisticRegression()
    log_model2 = LogisticRegression().fit(X2, Y)


    # Estimate our three coefficients
    regr_coef2 = log_model2.coef_.ravel()
    regr_coef2_tuple = tuple(value for value in regr_coef2)
    with open(path_result + "results.txt", "a") as outfile:
        newline = ("dist2indiff_excluded:\n\n" +
                   "regr_coef:\n" +
                   ("\tintercept:\t%s\n" % regr_coef2_tuple[0]) +
                   ("\tparam_gain:\t%s\n" % regr_coef2_tuple[1]) +
                   ("\tparam_loss:\t%s\n\n" % regr_coef2_tuple[2]))
        outfile.write(newline)


    # Compute Lambda and save its value to `lambda.txt`
    Lambda2 = -regr_coef2[2] / regr_coef2[1]
    with open("results/lambda_no_dist2indiff.txt", "a") as outfile:
        newline = (run + "\t" + subject + "\t" + str(Lambda2) + "\n")
        outfile.write(newline)


    # Calculate the new model's rate of misclassification
    probability_estimates2 = log_model2.predict_proba(X2)
    predictions2 = np.zeros(num_rows)
    predictions2[probability_estimates2[:, 1] > 0.5] = 1
    misclassification_rate2 = np.sum(predictions2 != Y) / num_rows
    with open(path_result + "results.txt", "a") as outfile:
        newline = ("misclass_rate:\t%s\n\n" % misclassification_rate2)
        outfile.write(newline)

    # Another Wald test to assess the statistical significance of our two
    # regressors, without the distance to indifference
    p_values2 = waldtest(X2, regr_coef2, probability_estimates2)
    p_values2_tuple = tuple(value for value in p_values2)
    with open(path_result + "results.txt", "a") as outfile:
        newline = ("stat_signif:\n" +
                   ("\tintercept:\t%s\n" % p_values2_tuple[0]) +
                   ("\tparam_gain:\t%s\n" % p_values2_tuple[1]) +
                   ("\tparam_loss:\t%s\n" % p_values2_tuple[2]))
        outfile.write(newline)
