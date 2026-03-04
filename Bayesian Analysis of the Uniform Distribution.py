# Imports

import argparse
import time
import matplotlib # type: ignore # type: ignor
import numpy as np # type: ignore

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt # type: ignore


def rand_uniform(a, b):
    # TODO: Return a random number x ~ U(a, b)

    return np.random.uniform(a, b)


def pareto_pdf(x, start_x, shape):
    """Pareto probability density function"""

    return (shape * start_x**shape / x**(shape + 1)) if x >= start_x else 0


def init():

    # The true distribution of the population U(0, theta_pop), where theta_pop is unknown

    theta_pop = float(args.theta_pop)

    # The prior distribution Pa(theta ; theta_prior, shape_prior)

    theta_prior = float(args.theta_prior)
    shape_prior = float(args.shape_prior)

    # Sampled observations

    samples_pop = []

    # Init plot

    ax = plt.gca()

    # Sampling loop

    for i in range(25):
        # Clear previous plot

        plt.cla()

        # Add a new sample to the observations

        samples_pop.append(rand_uniform(0, theta_pop))

        # Compute posterior parametrization

        N = len(samples_pop)
        theta_hat = max(samples_pop)
        theta_post = max(theta_prior, theta_hat)
        shape_post = shape_prior + N

        # Plot samples from the population

        ax.hist(samples_pop, density=True, label="Samples from population")

        # Plot posterior

        x = [
            args.plot_boundaries[0]
            + i
            * (args.plot_boundaries[1] - args.plot_boundaries[0])
            / args.plot_resolution
            for i in range(args.plot_resolution)
        ]
        y = [pareto_pdf(xi, theta_post, shape_post) for xi in x]
        ax.plot(x, y, label="Posterior Pa(theta | c, N + K)")

        # Redraw

        ax.legend(loc="upper right")
        plt.axis(args.plot_boundaries)
        plt.title(args.title)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.5)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments

    parser.add_argument(
        "--title",
        default="Ex2: Bayesian analysis of the Uniform Distribution",
        required=False,
    )

    parser.add_argument("--theta-pop", default=500, required=False)

    parser.add_argument("--theta-prior", default=0, required=False)

    parser.add_argument("--shape-prior", default=0, required=False)

    parser.add_argument("--plot-resolution", default=100, required=False)

    parser.add_argument(
        "--plot-boundaries",
        default=[-5, 600, 0, 0.05],  # min_x, max_x, min_y, max_y
        required=False,
    )

    parser.add_argument("--font-size", default=10, required=False)

    args = parser.parse_args()

    init()
