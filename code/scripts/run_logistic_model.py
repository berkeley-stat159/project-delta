"""Fitting logistic model using sklearn package

Goal:
-----
	1. Getting esitmated coeffiecients for each features(with or without euclidean distance)
	2. Hypothesis test for coeffiecients using z-values

Steps:
------
1. construct design matrix
2. create a LogisticRegression object to create a model
3. fit a logistic model using training data
4. call coef_ method to get coeffiecients
5. Calculate misclassification rate for training data
6. Wald test to test the significance of the coeffiecients

"""

from __future__ import absolute_import, division, print_function
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import norm
import sys
import numpy.linalg as npl
sys.path.append("code/utils")
from make_class import *

### With Euclidean Distance

# Step1: read files and construct design matrix
X = run('001','001').design_matrix()
Y = run('001','001').behav[:, 5]

# Step 2: create LogisticRegression instance
log_mod = LogisticRegression()

# Step 3: fit training data
log_mod.fit(X,Y)

# Step 4: coeffiecients
beta_hat = log_mod.coef_
beta_hat = beta_hat.ravel()
print("The esitmated coeffiecients are {}".format(beta_hat))
lam = -beta_hat[2]/beta_hat[1]
print("The Behavioral loss aversion (lambda) is {}".format(lam))

# Step 5: Training accuracy
N = X.shape[0]
prob = log_mod.predict_proba(X)
pred = np.zeros(N)
pred[prob[:,1]>0.5] = 1
misclass_rate = np.sum(pred!=Y)/N
print("The misclassification rate is {}".format(misclass_rate))
# Step 6: Wald test
var = np.diag(N*np.product(prob,axis=1))
S = npl.inv(X.T.dot(var.dot(X)))
SE = np.sqrt(np.diagonal(S))
z = beta_hat/SE
p_val = (1-norm.cdf(abs(z)))*2
print("The p values for each coeffiecient are {}".format(p_val))
print("================================================================================\n")

#==============================================================================
### Without Euclidean Distance

# Step1: read files and construct design matrix
X2 = run('001','001').design_matrix(euclidean_dist = False)

# Step 2: create LogisticRegression instance
log_mod2 = LogisticRegression()

# Step 3: fit training data
log_mod2.fit(X2,Y)

# Step 4: coeffiecients
beta_hat2 = log_mod2.coef_
beta_hat2 = beta_hat2.ravel()
print("The esitmated coeffiecients (without euclidean distance) are {}".format(beta_hat2))
lam2 = -beta_hat2[2]/beta_hat2[1]
print("The Behavioral loss aversion (lambda) is {}".format(lam2))

# Step 5: Training accuracy
prob2 = log_mod2.predict_proba(X2)
pred2 = np.zeros(N)
pred2[prob2[:,1]>0.5] = 1
misclass_rate2 = np.sum(pred2!=Y)/N
print("The misclassification rate is {}".format(misclass_rate2))

# Step 6: Wald test
var2 = np.diag(N*np.product(prob2,axis=1))
S2 = npl.inv(X2.T.dot(var2.dot(X2)))
SE2 = np.sqrt(np.diagonal(S2))
z2 = beta_hat2/SE2
p_val2 = (1-norm.cdf(abs(z2)))*2
print("The p values for each coeffiecient are {}".format(p_val2))



