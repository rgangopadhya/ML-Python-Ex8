import numpy as np
import matplotlib.pyplot as pyplot

def estimateGaussian(X):
	"""
	Compute, for each feature in X, the mean and variance (parameters of 
		a Gaussian distribtion)
	X has m rows (examples), n columns (features), so mu and var are n x 1
	"""

	mu = np.mean(X, 0)
	var = np.var(X, 0)
	return (mu, var)

def multivariateGaussian(X, mu, var):
	"""
	Computes PDF of multivariate gaussian
	if var is a matrix, it is interpreted as a covariance matrix
	if var is a vector, then it is interpreted as the diagonal entries of a (diagonal)
	covariance matrix
	"""	
	import math
	k = mu.shape[0]
	if len(var.shape) == 1:
		var = np.diag(var)

	X = X - mu
	p = (2 * math.pi) ** (-k / 2) * np.linalg.det(var) ** (-0.5) *\
		math.exp(-0.5 * np.sum(np.transpose(X) * linalg.inv(var) * X, 1))	