"""
Logistic Model

"""
from __future__ import division, print_function, absolute_import
import numpy as np 
import numpy.linalg as npl
from scipy.optimize import fmin_bfgs

def logistic(z):
	"""logistic function, return probability that the response is 1"""
	return np.exp(z) / (1.0 + np.exp(z))

def predict(beta, x):
	"""return the predicted class -1 or 1

	Parameter:
	---------
	beta: an array of regression coefficients
	x: an one-dimensional array of train data, same length with beta
	
	Return:
	------
	1 if the probability is greater than 0.5
	-1 otherwise
	"""
	if logistic(np.dot(beta, x)) > 0.5:
		return 1
	return -1

def log_likelihood(X, Y, beta, lam=0.1):
	"""calculate log likelihood, where we have L2 penalty term
	and lam is tunning parameter lambda

	Parameters:
	----------
	X: train data matrix (2-dimensional array)
	Y: an array, response
	beta: an array of regression coefficients
	lam: tunning parameter, control the strength of penalty term

	Return:
	------
	log_likelihood
	"""
	return np.sum(np.log(logistic(Y * np.dot(X, beta)))) - lam/2 * np.dot(beta, beta)

def log_likelihood_grad(X, Y, beta, lam=0.1):
	"""return gradients for log likelihood functions respect to each beta
	
	Parameters:
	-----------
	X: 2D array
		design matrix
	Y: 1D array
		response
	beta: an array 
		regression coefficients
	lam: tunning parameter

	Return:
	------
	s: an array
		gradients for log likelihood functions respect to each beta
	"""
	s = np.zeros(len(beta))
	for i in range(X.shape[0]):
		s += Y[i] * X[i,:] * logistic(-Y[i] * np.dot(X[i,:], beta))
	s -= lam * beta
	return s

def train_beta_hat(X, Y, lam=0.1):
	"""return the beta hat that maximize log likelihood

	Parameters:
	----------
	X: 2-D array
		design matrix
	Y: 1-D array
		response
	lam: tunning parameter

	Return:
	------
	an array
		the beta hats that maximize log likelihood
	"""
	def f(beta):
		return -log_likelihood(X, Y, beta, lam)
	def fprime(beta):
		return -log_likelihood_grad(X, Y, beta, lam)
	initial = np.zeros(X.shape[1])
	return fmin_bfgs(f, initial, fprime, disp=False)

def accuracy(X, Y, beta):
	"""return accuracy of logistic regression

	Parameters:
	----------
	X: 2-D array
		design matrix
	Y: 1-D array
		response
	beta: 1-D array
		regression coefficients

	Return:
	------
	accuracy of the model
	"""
	n_correct = 0
	for i in range(len(Y)):
		if predict(beta, X[i,:]) == Y[i]:
			n_correct += 1
	return n_correct*1.0/ len(Y)

def fold(arr, K, i):
	"""Make one fold. return test set and train set

	Parameters:
	-----------
	arr: a 2-D array
	K: the number of folds
	i: index of folds that is seleced 

	Return:
	------
	a tuple: the first is the heldout test data, the second is the rest train data
	"""
	N = len(arr)
	size = np.ceil(1.0 * N / K)
	arange = np.arange(N)
	heldout = np.logical_and(i * size <= arange, arange < (i+1) * size)
	rest = np.logical_not(heldout)
	return arr[heldout], arr[rest]

def kfold(arr, K):
	"""Split to K folds

	Parameters:
	-----------
	arr: 2-D array
		full-set train data
	K: number of folds

	Return:
	------
	a list of tuples, which is return by fold function
	"""
	return [fold(arr, K, i) for i in range(K)]

def cv_accuracy(all_X, all_Y, lam):
	"""return the average accuracy for k-fold CV
	
	Parameters:
	----------
	all_X: 
		a list of tuples, each tuple contains a test data (2D array) and train data (2D array) for each fold
	all_Y:
		a list of tuples, each tuple contains a test response (1D array) and train response (1D array) for each fold
	lam:
		tunning parameter
	
	Return:
	------
	average accuracy over all K folds
	"""
	s = 0
	K = len(all_X)
	for i in range(K):
		X_heldout, X_rest = all_X[i]
		Y_heldout, Y_rest = all_Y[i]
		beta = train_beta_hat(X_rest, Y_rest, lam)
		s += accuracy(X_heldout, Y_heldout, beta)
	return s*1.0/ K

def train_lambda(X, Y, K=10):
	"""Use K-fold CV to find the best lambda
	
	Parameters:
	----------
	X: train data
	Y: train response
	K: number of folds

	Return:
	------
	the best lambda that causes the highest CV accuracy
	"""
	all_lambda = np.arange(0, 1, 0.1)
	all_X = kfold(X, K)
	all_Y = kfold(Y, K)
	all_acc = np.array([cv_accuracy(all_X, all_Y, lam) for lam in all_lambda])
	return all_lambda[all_acc.argmax()]

