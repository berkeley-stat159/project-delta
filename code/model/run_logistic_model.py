from __future__ import absolute_import, division, print_function
from logistic_model import *

arr = read_files("./temp_data_for_testing/behavdata.txt")
X, Y = construct_mat(arr)

bestlam = train_lambda(X, Y, K=10)
beta_hat = train_beta_hat(X, Y, lam=bestlam)
temp_data_for_testinga_gain = beta_hat[1]
beta_loss = beta_hat[2]

behav_loss_aversion = -beta_loss/beta_gain