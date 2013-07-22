import numpy as np
import matplotlib.pyplot as pyplot

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambd):
	"""
	Collaborative filtering cost function
	"""
	#unfold X and theta from params
	X = np.reshape(params[:num_movies, :], (num_movies, num_features))
	Theta = np.reshape(params[num_movies:, :], (num_users, num_features))
	#compute the estimated rating for each user and each movie
	Y_est = np.dot(X, np.transpose(Theta))
	#compute the difference in rating and prediction for movies that were rated
	Y_err = (Y_est - Y) * R

	#cost function
	J = (1.0 / 2) * np.sum(Y_err * Y_err) + (lambd / 2.0) * (
		np.sum(np.diag(np.dot(np.transpose(X), X))) +\
		 np.sum(np.diag(np.dot(np.transpose(Theta), Theta))))
	x_grad = np.dot(Y_err, Theta) + (lambd * X) 
	Theta_grad = np.dot(np.transpose(Y_err), X)\
	 + (lambd * Theta)
	grad = np.vstack((x_grad, Theta_grad))
	return (J, grad)

def computeNumericalGradient(J, theta):
	"""
	Computes gradient at a point by perturbing by a small amount and 
	looking at the slope
	"""
	#there has to be a clean way to vectorize this..
	numgrad = np.zeros(theta.shape)
	perturb = np.zeros(theta.shape)
	e = 1e-4
	for i in xrange(theta.shape[0]):
		for j in xrange(theta.shape[1]):
			perturb[i, j] = e
			loss1 = J(theta - perturb)[0]
			loss2 = J(theta + perturb)[0]
			numgrad[i, j] = (loss2 - loss1) / (2 * e)
			perturb[i, j] = 0
	return numgrad

def checkCostFunction(lambd = 0):
	"""
	Create collaborative filtering problem to check cost function and gradients
	"""
	#generate random matrices to test
	X_t =  np.random.rand(4, 3)
	Theta_t = np.random.rand(5, 3)

	#create Y and R, remove half of Y
	Y = np.dot(X_t, np.transpose(Theta_t))
	Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
	R = np.zeros(Y.shape)
	R[Y != 0] = 1

	X = np.random.normal(size = X_t.shape)
	Theta = np.random.normal(size = Theta_t.shape)
	
	num_users = Y.shape[1]
	num_movies = Y.shape[0]
	num_features = Theta_t.shape[1]

	#calculate numeric gradient and cofiCost gradient, for comparison
	params = np.vstack((X, Theta))
	numgrad = computeNumericalGradient(lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambd), params)
	(cost, grad) = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambd)

	print np.vstack((numgrad.flatten(), grad.flatten()))
	print "The difference is ", np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)			

def loadMovieList(filename):
	"""
	Reads fixed movie list and returns list of the names
	"""
	import re	
	f = open(filename)
	movie_list = [re.sub('[0-9()\\n]','',line).strip() for line in f]
	f.close()
	return movie_list

def normalizeRatings(Y, R):
	"""
	Subtract mean rating for every movie (row)
	"""
	Ymean = np.zeros((Y.shape[0], 1))
	Ynorm = np.zeros(Y.shape)
	for i in xrange(Y.shape[0]):
		idx = R[i, :] == 1
		Ymean[i] = np.mean(Y[i, idx])
		Ynorm[i, idx] = Y[i, idx] - Ymean[i]
	return (Ynorm, Ymean)	
