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

	X = X -mu
	p = (2 * math.pi) ** (-k / 2) * np.linalg.det(var) ** (-0.5) *\
		np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(var)) * X, 1))	
	return p

def visualizeFit(X, mu, var):
	"""
	Visualize dataset with distribution (plot contours)
	"""
	import matplotlib.cm as cm	

	l = np.arange(0, 35, 0.5)
	X1, X2 = np.meshgrid(l, l)
	Z = multivariateGaussian(np.transpose(np.vstack((X1.flatten('C'), X2.flatten('C')))), mu, var)
	Z = np.reshape(Z, X1.shape, 'C')
	pyplot.plot(X[:, 0], X[:, 1], 'bx')
	pyplot.contour(X1, X2, Z, np.power(10.0, np.arange(-20, 1, 3)), cmap=cm.RdBu)			

def selectThreshold(yval, pval):
	"""
	Find the best threshold to use for identifying anomalies, based on
	finding threshold which gives us the best F1 score
	"""
	bestEpsilon = 0
	bestF1 = 0
	stepsize = (np.max(pval) - np.min(pval)) / 1000.0

	for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
		#for each epsilon, determine the F1 score
		#F1 is the harmonic mean of the precision and recall
		TP = np.sum(pval[yval == 1] < epsilon)
		FP = np.sum(pval[yval == 0] < epsilon)
		FN = np.sum(pval[yval == 1] >= epsilon)
		TN = np.sum(pval[yval == 0] >= epsilon)

		if (TP + FP) > 0: prec = (1.0 * TP) / (TP + FP)
		else: prec = 0
		if (TP + FN) > 0: rec = (1.0 * TP) / (TP + FN)
		else: prec = 0
		if (prec + rec) > 0: F1 = prec * rec / (prec + rec)
		else: F1 =0
		if F1 > bestF1:
			bestEpsilon = epsilon
			bestF1 = F1
	return (bestEpsilon, bestF1)		