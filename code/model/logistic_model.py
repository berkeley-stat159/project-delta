from __future__ import division, print_function, absolute_import
import numpy as np 
import numpy.linalg as npl
from scipy.optimize import fmin_bfgs

def read_files(filename):
	"""read from behav.txt and covert data to array format"""
	with open(filename,"r") as infile:
		lines = infile.readlines()
	lines = lines[1:]
	val = [line.split() for line in lines]
	val_arr = np.array(val, dtype=float)
	return val_arr

def construct_mat(array):
	"""Construct the design matrix using the array return from the
	read_files function.
	"""
	mat = array[:,[1,2,5]]
	mat = mat[mat[:,2]!=-1,:]
	design_matrix = np.ones(mat.shape)
	design_matrix[:,1:] = mat[:,:2]
	response = mat[:,2]
	return design_matrix, response

def logistic(z):
	"""logistic function, return probability that the response is 1"""
	return np.exp(z) / (1.0 + np.exp(z))

def predict(beta, x):
	"""return the predicted class 0 or 1"""
	if logistic(np.dot(beta, x)) > 0.5:
		return 1
	return 0

def log_likelihood(X, Y, beta, lam=0.1):
	"""calculate log likelihood, where we have L2 penalty term
	and lam is tunning parameter lambda"""
	return np.sum(np.log(logistic(Y * np.dot(X, beta)))) - lam/2 * np.dot(beta, beta)

def log_likelihood_grad(X, Y, beta, lam=0.1):
	"""return gradients for log likelihood functions respect 
	to each beta"""
	s = np.zeros(len(beta))
	for i in range(X.shape[0]):
		s += Y[i] * X[i,:] * logistic(-Y[i] * np.dot(X[i,:], beta))
	s -= lam * beta
	return s

def train_beta_hat(X, Y, lam=0.1):
	"""return the beta hat that maximize log likelihood"""
	def f(beta):
		return -log_likelihood(X, Y, beta, lam)
	def fprime(beta):
		return -log_likelihood_grad(X, Y, beta, lam)
	initial = np.zeros(X.shape[1])
	return fmin_bfgs(f, initial, fprime, disp=False)

def accuracy(X, Y, beta):
	"""return accuracy of logistic regression"""
	n_correct = 0
	for i in range(len(Y)):
		if predict(beta, X[i,:]) == Y[i]:
			n_correct += 1
	return n_correct*1.0/ len(Y)

def fold(arr, K, i):
	"""Make one fold. return test set and train set"""
	N = len(arr)
	size = np.ceil(1.0 * N / K)
	arange = np.arange(N)
	heldout = np.logical_and(i * size <= arange, arange < (i+1) * size)
	rest = np.logical_not(heldout)
	return arr[heldout], arr[rest]

def kfold(arr, K):
	"""Split to K folds"""
	return [fold(arr, K, i) for i in range(K)]

def cv_accuracy(all_X, all_Y, lam):
	"""return the average accuracy for k-fold CV"""
	s = 0
	K = len(all_X)
	for i in range(K):
		X_heldout, X_rest = all_X[i]
		Y_heldout, Y_rest = all_Y[i]
		beta = train_beta_hat(X_rest, Y_rest, lam)
		s += accuracy(X_heldout, Y_heldout, beta)
	return s*1.0/ K

def train_lambda(X, Y, K=10):
	"""Use K-fold CV to find the best lambda"""
	all_lambda = np.arange(0, 1, 0.1)
	all_X = kfold(X, K)
	all_Y = kfold(Y, K)
	all_acc = np.array([cv_accuracy(all_X, all_Y, lam) for lam in all_lambda])
	return all_lambda[all_acc.argmax()]

