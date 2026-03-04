# Imports

import argparse
import random
import numpy as np # type: ignore
import matplotlib # type: ignore

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # type: ignore


def init():

	# Variables

	n = int(args.n)
	d = int(args.polynomial_order)
	res = args.plot_resolution

	# Generate Data

	x, y = data_generator(n, third_order_function)

	# Polynomial regression

	beta, sigma = polynomial_regression(x, y, d)

	# Evaluate regression function

	px = [min(x) + i * (max(x) - min(x)) / res for i in range(res)]
	py = polynomial_value(px, beta)

	# Plot

	plot([x, y], [px, py])


def third_order_function(x):
	res = x ** 3 - x
	return res


def polynomial_value(x, beta):
	"""Evaluate polynomial at data points X, given weight parameters beta
	"""
	polynomialValue = [sum([b * (xi ** j) for j, b in enumerate(beta)]) for i, xi in enumerate(x)]
	return polynomialValue


def data_generator(n, f):
	"""Generate training data
	"""

	x = [3 * (random.random() - 0.5) for _ in range(n)]
	y = [f(x[i]) + 2 * (random.random() - 0.5) for i in range(len(x))]

	return x, y


def polynomial_regression(x, y, d):
	"""Polynomial regression
	"""

	X = np.vander(x, d, increasing=True)  
	beta = np.linalg.lstsq(X, y, rcond=None)[0]  # solve for beta using least squares
	residual = np.array(y) - X @ beta  # compute residuals
	sigma = np.mean(residual**2)

	return beta.tolist(), sigma


def plot(observations, poly_estimate):
	"""Plot data and regression
	"""

	plt.rc('xtick')
	plt.rc('ytick')
	plt.figure(figsize = (4.8, 4))
	plt.title(args.title)
	plt.axis(args.plot_boundaries)
	plt.plot(poly_estimate[0], poly_estimate[1], color = 'red')
	plt.scatter(observations[0], observations[1], s = args.scatter_size)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show(block = True)
	plt.interactive(False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex4: Polynomial Regression',
						required = False)

	parser.add_argument('--n',
						default = 50,
						required = False)

	parser.add_argument('--polynomial-order',
						default = 4,
						required = False)

	parser.add_argument('--plot-boundaries',
						default = [-1.5, 1.5, -1.5, 3],  # min_x, max_x, min_y, max_y
						required = False)

	parser.add_argument('--plot-resolution',
						default = 100,
						required = False)

	parser.add_argument('--scatter-size',
						default = 20,
						required = False)

	args = parser.parse_args()

	init()
