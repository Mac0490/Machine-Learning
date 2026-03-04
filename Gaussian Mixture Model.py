# Imports

import argparse
import time
import numpy as np # type: ignore
import matplotlib # type: ignore

from scipy import stats # type: ignore

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt # type: ignore


def generate_data(amount, mean, cov):
    """Generate labeled data from two gaussian distributions"""
    rng = np.random.default_rng()
    x0 = rng.multivariate_normal(mean[0], cov[0], amount[0])
    x1 = rng.multivariate_normal(mean[1], cov[1], amount[1])

    y0 = np.ones(amount[0]) * 1
    y1 = np.ones(amount[1]) * 2

    data = np.concatenate((x0, x1), axis=0)
    labels = np.concatenate((y0, y1), axis=0)

    return data, labels


def get_density_meshgrid(mean, cov, grid_range=6):
    N = 100
    X = np.linspace(mean[0] - grid_range, mean[0] + grid_range, N)
    Y = np.linspace(mean[1] - grid_range, mean[1] + grid_range, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mean, cov)
    Z = rv.pdf(pos)

    return X, Y, Z


def init():
    # 2-dimensional Gaussian distribution parameters

    K = 2
    mean = np.array([[-2, -1], [1, 1]]) * 1
    cov = np.array([[[1, 0.8], [0.8, 1]], [[1, -0.4], [-0.4, 1]]])

    # Input data and labels of size n

    amount = np.array([40, 80])
    data, labels = generate_data(amount, mean, cov)

    # Initial model parameters

    x = data
    N = len(x)

    pz = np.ones(K) / K

    mean_hat = np.array([[-4, 4], [4, -4]], dtype="float")
    cov_hat = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype="float")

    # EM loop

    converged = False
    log_likelihood_list = []
    min_iter = 10
    n_iter = 0

    while not converged:
    # E-step
        resp = np.zeros((N, K))
        for k in range(K):
            rv = stats.multivariate_normal(mean_hat[k], cov_hat[k])
            resp[:, k] = pz[k] * rv.pdf(x)
        resp_sum = np.sum(resp, axis=1, keepdims=True)
        resp /= resp_sum

    # M-step
        Nk = np.sum(resp, axis=0)
        pz = Nk / N
        for k in range(K):
            mean_hat[k] = np.sum(resp[:, k, np.newaxis] * x, axis=0) / Nk[k]
        for k in range(K):
            x_centered = x - mean_hat[k]
            cov_hat[k] = (resp[:, k, np.newaxis] * x_centered).T @ x_centered / Nk[k]

    # Log-likelihood
        log_likelihood = 0
        for i in range(N):
            px_i = 0
            for k in range(K):
                rv = stats.multivariate_normal(mean_hat[k], cov_hat[k])
                px_i += pz[k] * rv.pdf(x[i])
            log_likelihood += np.log(px_i + 1e-10)

        print("Iteration {} - log-likelihood {:.2f}".format(n_iter, log_likelihood))
        log_likelihood_list.append(-1 * log_likelihood)
        n_iter += 1

        if n_iter > min_iter:
            diff = abs(
                log_likelihood_list[n_iter - min_iter - 1]
                - log_likelihood_list[n_iter - 1]
            )
            if diff < 0.1:
                converged = True
                print("Converged")

        # Plotting section (unchanged)
        plt.cla()
        plt.scatter(data[:, 0], data[:, 1], s=12, c=labels)
        cx0, cy0, cz0 = get_density_meshgrid(mean_hat[0], cov_hat[0])
        cx1, cy1, cz1 = get_density_meshgrid(mean_hat[1], cov_hat[1])
        plt.contour(cx0, cy0, cz0, [np.amax(cz0) * 0.25], alpha=0.5)
        plt.contour(cx1, cy1, cz1, [np.amax(cz1) * 0.25], alpha=0.5)
        plt.axis(args.plot_boundaries)
        plt.title(args.title)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.10)

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments

    parser.add_argument(
        "--title", default="Ex11: Gaussian Mixture Models", required=False
    )

    parser.add_argument("--plot-resolution", default=100, required=False)

    parser.add_argument(
        "--plot-boundaries",
        default=[-10, 10, -10, 10],  # min_x, max_x, min_y, max_y
        required=False,
    )

    args = parser.parse_args()

    init()
