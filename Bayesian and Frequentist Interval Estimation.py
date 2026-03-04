# Imports

import argparse
import math
import random
import matplotlib # type: ignore
import numpy as np # type: ignore
import time
import matplotlib.pyplot as plt # type: ignore
from scipy.stats import norm # type: ignore

matplotlib.use('TkAgg')


def rand_gaussian(mu, sigma, n = 0):

	def draw_sample():
		drawS = np.random.normal(mu, sigma, n) if n > 1 else np.random.normal(mu, sigma)
		return drawS

	if n > 0:
		return [draw_sample() for _ in range(n)]

	return draw_sample()


def normal_pdf(x, mu, sigma):
	normalPDF = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
	return normalPDF


def sample_mean(x):

	return np.mean(x)


def sample_std(sigma, n):
	sampleSTD = sigma / math.sqrt(n)
	return sampleSTD


def inv_cdf(area):

	invCDF = norm.ppf(1 - area / 2)
	return invCDF


def posterior_parametrization(x, sigma_pop, mu_prior, sigma_prior):

	mean_post = len(x)
	sample_mean_x = sample_mean(x)

	precision_posterior = (mean_post / sigma_pop**2) + (1 / sigma_prior**2)
	sigma_post = math.sqrt(1 / precision_posterior)
	mean_post = (mean_post * sample_mean_x / sigma_pop**2 + mu_prior / sigma_prior**2) / precision_posterior

	return mean_post, sigma_post


def CI(mean, err, z):

	lower_bound = mean - z * err
	upper_bound = mean + z * err

	return [lower_bound, upper_bound]


def init():

	# Variables

	res = args.plot_resolution

	# Inverse cumulative density function

	z = inv_cdf(args.alpha)

	# The true distribution of the population: N(mu_pop, sigma_pop ** 2)
	# Assume that sigma is known and mu is unknown.

	mu_pop = args.mu_pop
	sigma_pop = args.sigma_pop

	# Prior distribution of mu ~ N(mu_prior, sigma_prior ** 2)

	mu_prior = args.mu_prior
	sigma_prior = args.sigma_prior

	# Sampled observations

	samples_pop = []

	# Init plot

	ax = plt.gca()

	# Sampling loop

	for i in range(500):
		# Clear previous plot

		plt.cla()

		# Add a new sample to the observations

		samples_pop.append(rand_gaussian(mu_pop, sigma_pop))

		# Compute sample mean and standard deviation

		s_mean = sample_mean(samples_pop)
		s_std = sample_std(sigma_pop, len(samples_pop))

		# Compute the frequentist interval

		ci = CI(s_mean, s_std, z)

		# Compute posterior parametrization

		mu_posterior, sigma_posterior = posterior_parametrization(samples_pop, sigma_pop, mu_prior, sigma_prior)

		# Compute the credible interval

		posterior_ci = CI(mu_posterior, math.sqrt(sigma_posterior), z)

		# Plot samples from the population

		ax.hist(samples_pop, density = True, label = 'Samples from population')

		# Plot the true mean

		ax.axvline(x = mu_pop, label = 'pPopulation mu', color = 'r')

		# Plot frequentist interval

		ax.axvline(x = ci[0], label = 'Lower confidence interval', color = 'c')
		ax.axvline(x = ci[1], label = 'Upper confidence interval', color = 'c')

		# Plot prior

		min_x = mu_prior - sigma_prior * 2
		max_x = mu_prior + sigma_prior * 2
		x = [min_x + i * (max_x - min_x) / res for i in range(res)]
		y = [normal_pdf(xi, mu_prior, sigma_prior) for xi in x]
		ax.plot(x, y, label = 'Prior f(mu)')

		# Plot posterior

		min_x = mu_posterior - 1 * 4
		max_x = mu_posterior + 1 * 4
		x = [min_x + i * (max_x - min_x) / res for i in range(res)]
		y = [normal_pdf(xi, mu_posterior, sigma_posterior) for xi in x]
		ax.plot(x, y, label = 'Posterior f(x|mu)')

		# Redraw

		ax.legend(loc='upper right')
		plt.axis(args.plot_boundaries)
		plt.title(args.title)
		plt.draw()
		plt.pause(1e-17)
		time.sleep(0.001)

	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex1: Confidence and Credible Intervals',
						required = False)

	parser.add_argument('--alpha',
						default = 0.99,
						required = False)

	parser.add_argument('--mu_pop',
						default = 10,
						required = False)

	parser.add_argument('--sigma_pop',
						default = 3,
						required = False)

	parser.add_argument('--mu_prior',
						default = 14,
						required = False)

	parser.add_argument('--sigma_prior',
						default = 10,
						required = False)

	parser.add_argument('--plot-resolution',
						default = 100,
						required = False)

	parser.add_argument('--plot-boundaries',
						default = [4, 16, 0, 0.8],  # min_x, max_x, min_y, max_y
						required = False)

	parser.add_argument('--font-size',
						default = 10,
						required = False)

	args = parser.parse_args()

	init()

#As the number of samples increases, can you notice any changes? 
# As number of samples increases, both confidence intervals and credible intervals shrink, meaning we become more certain about the population mean. 

# What happens if the prior is uniformly distributed, e.g., the variance is very large?
#If the prior has a very large variance, it has little influence and the posterior behaves similarly to the frequentist approach, relying mainly on the data. #Over time, both methods converge to the true population mean.