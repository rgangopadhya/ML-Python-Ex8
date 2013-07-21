import numpy as np
import matplotlib.pyplot as pyplot

def cofiCostFunc(X, theta, Y, R, num_users, num_movies, num_features, lambd):
	"""
	Collaborative filtering cost function
	"""
	#compute the estimated rating for each user and each movie
	Y_est = np.dot(X, np.transpose(theta))
	#compute the difference in rating and prediction for movies that were rated
	Y_err = (Y_est - Y) * R
	#cost function
	J = (1 / 2) * np.sum(Y_err * Y_err) + (lambd / 2) * (
		np.sum(np.diag(np.dot(X, X))) + np.sum(np.diag(np.dot(theta, theta))))
	x_grad = np.dot(Y_err, theta) + (lambd * X) 
	theta_grad = np.dot(np.transpose(Y_err), X)\
	 + (lambd * theta)
	grad = np.vstack((x_grad, theta_grad))
	return (J, grad)