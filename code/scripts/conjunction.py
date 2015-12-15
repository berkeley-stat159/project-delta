"""
Conjunct Analysis Across 16 Subjects

Purpose
-------
To find the correspondence between neural and behavioral loss aversion.

Input
-----
The neural and behavioral loss aversion data produced in the logistic regression
script and the generalized linear model script.

Details
-------
1) Two figures for both behavioral loss aversion (with and without the distance
   from indifference).
2) Each figure contains three scatterplots and ordinary least squares (OLS)
   regression models (one for each run).
3) The horizontal axis is the neural loss aversion, and the vertical axis is the
   natural logarithm of the behavioral loss aversion.
4) Two regression lines are plotted: by OLS (black) or robust regression (blue).
5) Outliers identified by robust regression are indicated by red dots.

Output
------
- "correlation_dist2indiff", which includes as a regressor the distance from
  indifference
- "correlation_no_dist2indiff", which does not include as a regressor the
  distance from indifference
"""
from __future__ import absolute_import, division, print_function
from matplotlib import colors
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl


# Load the files containing the neural loss aversion data from other analyses
path = "results/"

infile = list(open(path + "lambda_euclidean_dist.txt"))[1:]
lambda_dist = np.array([row.split() for row in infile]).astype("float")
lambda_dist_run1 = lambda_dist[0:15, 2]
lambda_dist_run2 = lambda_dist[16:31, 2]
lambda_dist_run3 = lambda_dist[32:47, 2]

infile = list(open(path + "lambda_no_euclidean_dist.txt"))[1:]
lambda_no_dist = np.array([row.split() for row in infile]).astype("float")
lambda_no_dist_run1 = lambda_no_dist[0:15, 2]
lambda_no_dist_run2 = lambda_no_dist[16:31, 2]
lambda_no_dist_run3 = lambda_no_dist[32:47, 2]

infile = list(open(path + "neural_loss_aversion.txt"))[1:]
lambda_neural = np.array([row.split() for row in infile]).astype("float")
lambda_neural_run1 = lambda_neural[0:15, 2]
lambda_neural_run2 = lambda_neural[16:31, 2]
lambda_neural_run3 = lambda_neural[32:47, 2]


# We create a number of plots depicting the relationship between neural and
# behavioral loss aversion:


#####################################
# Distance to Indifference Included #
#####################################

t = np.arange(-2,3)
plt.figure(figsize=(18, 15), dpi=100)
plt.subplot(311)
title = ("Correspondence Between Neural and Beavioral Loss Aversion" +
         "(Distance to Indifference Included)")
plt.suptitle(title, fontsize=20)
plt.scatter(lambda_neural_run1, np.log(lambda_dist_run1), color="black")

# RUN 1
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run1[:,np.newaxis]
y = np.log(lambda_dist_run1)
regr.fit(X, y)

# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], ".g", label="Inliers")
plt.plot(X[outlier_mask], y[outlier_mask], ".r", label="Outliers")

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="Linear Regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="Robust Regressor")
plt.title("Run 1", fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")

plt.subplot(312)
plt.scatter(lambda_neural_run2, np.log(lambda_dist_run2), color="black")

# RUN 2
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run2[:, np.newaxis]
y = np.log(lambda_dist_run2)
regr.fit(X, y)

# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], ".g", label="Inliers")
plt.plot(X[outlier_mask], y[outlier_mask], ".r", label="Outliers")

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 2",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")

plt.subplot(313)
plt.scatter(lambda_neural_run3, np.log(lambda_dist_run3), color="black")

# RUN 3
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run3[:,np.newaxis]
y = np.log(lambda_dist_run3)
regr.fit(X, y)

# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], ".g", label="Inliers")
plt.plot(X[outlier_mask], y[outlier_mask], ".r", label="Outliers")

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 3",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")
plt.legend(loc="lower right")
plt.savefig(path + "correlation_dist2indiff")
plt.close()


#####################################
# Distance to Indifference Excluded #
#####################################

plt.figure(figsize=(18, 15), dpi=100)
plt.subplot(311)
plt.scatter(lambda_neural_run1, np.log(lambda_no_dist_run1),color="black")
title = ("Correspondence Between Neural and Behavioral Loss Aversion" +
	     "(Distance to Indifference Excluded)")
plt.suptitle(title, fontsize=20)

# RUN 1
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run1[:,np.newaxis]
y = np.log(lambda_no_dist_run1)
regr.fit(X, y)

# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], ".g", label="Inliers")
plt.plot(X[outlier_mask], y[outlier_mask], ".r", label="Outliers")

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 1",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")

plt.subplot(312)
plt.scatter(lambda_neural_run2, np.log(lambda_no_dist_run2),color="black")

# RUN 2
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run2[:,np.newaxis]
y = np.log(lambda_no_dist_run2)
regr.fit(X, y)

# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], ".g", label="Inliers")
plt.plot(X[outlier_mask], y[outlier_mask], ".r", label="Outliers")

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 2",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")

plt.subplot(313)
plt.scatter(lambda_neural_run3, np.log(lambda_no_dist_run3),color="black")

# RUN 3
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run3[:,np.newaxis]
y = np.log(lambda_no_dist_run3)
regr.fit(X, y)

# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], ".g", label="Inliers")
plt.plot(X[outlier_mask], y[outlier_mask], ".r", label="Outliers")

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")

plt.title("Run 3",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")
plt.legend(loc="lower right")
plt.savefig(path + "correlation_no_dist2indiff")
plt.close()
