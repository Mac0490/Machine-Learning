# Imports

import argparse
import datetime
import time
import math
import numpy as np # type: ignore
import matplotlib # type: ignore

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # type: ignore

def rand_gaussian(mu, sigma, n = 0):
	def draw_sample():
		return np.random.normal(mu, sigma)

	if n > 0:
		return [draw_sample() for _ in range(n)]

	return draw_sample()


def generate_data(n, data, labels, param, label):
	"""Generate labeled data from a gaussian distributions
	"""

	mu = param[0]
	sigma = param[1]

	for i in range(n):

		x1 = rand_gaussian(mu[0], sigma[0])
		x2 = rand_gaussian(mu[1], sigma[1])
		data_point = [1, x1, x2]
		data.append(data_point)
		labels.append(label)

		#print("Data shape:", np.array(data).shape)
		#print("First few data points:", data[:5])

	return data, labels



def linear_discriminant(x, w):
	"""Linear discriminant function
	"""
	l = sum([w[i]*x[i] for i in range(len(w))])
	return l


def update_parameter(w, learning_rate, label, x):
	"""Update model parameters
	"""
	p = w + learning_rate * label * x
	return p


def decision_boundary(x, w):
	"""Decision boundary function
	"""
#-(w[0] + w[1] * x) / w[2]
	return -(w[0] + w[1] * x) / w[2]


def init():

	# 2d distribution parameters: [[mean_0, mean_1], [sigma_0, sigma_1]],
	# with a diagonal covariance matrix.

	param_0 = [[30, 40], [10, 15]]
	param_1 = [[-10, -5], [10, 6]]

	# Generate input data and labels of size n

	data, labels = generate_data(30, [], [], param_0, 1)
	data, labels = generate_data(20, data, labels, param_1, -1)

	# Initial model parameters

	w = [100, -10, 20]

	# Learning rate

	learning_rate = 1.0

	# Init plot

	ax = plt.gca()

	# Perceptron training loop

	converged = False

	while not converged:

		# Clear previous plot

		plt.cla()

		# Iterate the data

		converged = True

		for x, label in zip(data, labels):

			error = -label * linear_discriminant(x, w)

			if error > 0:

				# Update all model parameters w

				w = [update_parameter(w[i], learning_rate, label, x[i]) for i in range(len(x))]

				converged = False

		# Plot all observations

		plt.scatter(np.array(data)[:, 1], np.array(data)[:, 2], s = 12, c = labels)
		#plt.scatter(np.array(data)[:, 1], np.array(data)[:, 2], s=12, c=labels, cmap='coolwarm')

		# Plot decision boundary function

		x = [args.plot_boundaries[0] + i * (args.plot_boundaries[1] - args.plot_boundaries[0]) / args.plot_resolution for i in range(args.plot_resolution)]
		y = [decision_boundary(xi, w) for xi in x]
		ax.plot(x, y, label = 'Decision boundary function')

		# Redraw

		plt.axis(args.plot_boundaries)
		plt.title(args.title)
		plt.draw()
		plt.pause(1e-17)
		time.sleep(0.025)

	print('Training done')

	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex3: Perceptron Convergence Theorem',
						required = False)

	parser.add_argument('--plot-resolution',
						default = 100,
						required = False)

	parser.add_argument('--plot-boundaries',
						default = [-10, 10, -10, 10],  # min_x, max_x, min_y, max_y
						required = False)

	args = parser.parse_args()

	init()

