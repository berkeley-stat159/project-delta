from __future__ import division, print_function, absolute_import
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy.linalg as npl
from sklearn import linear_model
import numpy as np

file_path = "results/"
lambda_w_dist = np.array([row.split() for row in list(open(file_path+"lambda_euclidean_dist.txt"))[1:]]).astype("float")
lambda_wo_dist = np.array([row.split() for row in list(open(file_path+"lambda_no_euclidean_dist.txt"))[1:]]).astype("float")
lambda_neural = np.array([row.split() for row in list(open(file_path+"neural_loss_aversion.txt"))[1:]]).astype("float")

lambda_w_dist_run1 = lambda_w_dist[0:15, 2]
lambda_w_dist_run2 = lambda_w_dist[16:31, 2]
lambda_w_dist_run3 = lambda_w_dist[32:47, 2]

lambda_wo_dist_run1 = lambda_wo_dist[0:15, 2]
lambda_wo_dist_run2 = lambda_wo_dist[16:31, 2]
lambda_wo_dist_run3 = lambda_wo_dist[32:47, 2]

lambda_neural_run1 = lambda_neural[0:15, 2]
lambda_neural_run2 = lambda_neural[16:31, 2]
lambda_neural_run3 = lambda_neural[32:47, 2]

# Plotting
t = np.arange(-2,3)
plt.figure(figsize=(18, 15), dpi=100)
plt.subplot(311)
plt.suptitle("Correspondence between neural and behavioral loss aversion (Euclidean Distance Included)",fontsize=20)
plt.scatter(lambda_neural_run1, np.log(lambda_w_dist_run1),color='black')
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run1[:,np.newaxis]
y = np.log(lambda_w_dist_run1)
regr.fit(X, y)
# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 1",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")

plt.subplot(312)
plt.scatter(lambda_neural_run2, np.log(lambda_w_dist_run2),color='black')
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run2[:,np.newaxis]
y = np.log(lambda_w_dist_run2)
regr.fit(X, y)
# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 2",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")

plt.subplot(313)
plt.scatter(lambda_neural_run3, np.log(lambda_w_dist_run3),color='black')
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run3[:,np.newaxis]
y = np.log(lambda_w_dist_run3)
regr.fit(X, y)
# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 3",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")
plt.legend(loc='lower right')
plt.savefig(file_path + "correlation_v1")
plt.close()

plt.figure(figsize=(18, 15), dpi=100)
plt.subplot(311)
plt.scatter(lambda_neural_run1, np.log(lambda_wo_dist_run1),color='black')
plt.suptitle("Correspondence between neural and behavioral loss aversion (Euclidean Distance Excluded)",fontsize=20)
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run1[:,np.newaxis]
y = np.log(lambda_wo_dist_run1)
regr.fit(X, y)
# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 1",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")

plt.subplot(312)
plt.scatter(lambda_neural_run2, np.log(lambda_wo_dist_run2),color='black')
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run2[:,np.newaxis]
y = np.log(lambda_wo_dist_run2)
regr.fit(X, y)
# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")
plt.title("Run 2",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")

plt.subplot(313)
plt.scatter(lambda_neural_run3, np.log(lambda_wo_dist_run3),color='black')
# OLS
regr = linear_model.LinearRegression()
X = lambda_neural_run3[:,np.newaxis]
y = np.log(lambda_wo_dist_run3)
regr.fit(X, y)
# Robust
regr2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
regr2.fit(X, y)
inlier_mask = regr2.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')

plt.plot(t, regr.predict(t[:, np.newaxis]), "-k", label="linear regressor")
plt.plot(t, regr2.predict(t[:, np.newaxis]), "-b", label="robust regressor")

plt.title("Run 3",fontsize=16)
plt.xlabel("Neural Loss Aversion")
plt.ylabel("Behavioral Loss Aversion")
plt.legend(loc='lower right')
plt.savefig(file_path + "correlation_v2")
plt.close()

