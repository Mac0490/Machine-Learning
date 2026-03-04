# Imports

import argparse
import random
import time
import numpy as np  # type: ignore
import matplotlib  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # type: ignore

def init():
    n = int(args.n)
    sigma = 0.4**2
    tau = 5.0**2

    x = []
    y = []
    ax = plt.gca()

    for i in range(n):
        plt.cla()
        x_, y_ = data_generator(1, third_order_function, sigma)
        x.append(x_[0])
        y.append(y_[0])

        d = model_selection(x, y, sigma, tau)
        print(f"[Data points {i}/{n}] Selected polynomial order: {d}")

        beta_hat, covar = posterior(poly_expansion(x, d), y, sigma, tau)
        px = np.linspace(args.plot_boundaries[0], args.plot_boundaries[1], args.plot_resolution)
        pxx = poly_expansion(px, d)
        py, py_std = posterior_predictive(pxx, beta_hat, covar, sigma)

        upper = (py + py_std).flatten()
        lower = (py - py_std).flatten()

        plt.axis(args.plot_boundaries)
        plt.fill_between(px, upper, lower, alpha=0.5, label="Standard deviation")
        plt.scatter(x, y, s=args.scatter_size, label="Observations")
        plt.plot(px, py, color="red", label="Polynomial estimate")
        plt.plot(px, [third_order_function(xi) for xi in px], color="blue", label="Generating function")

        ax.legend(loc="upper right")
        plt.title(args.title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.draw()
        plt.pause(1e-17)
        time.sleep(1)

    plt.show()

def third_order_function(x):
    return x**3 - x

def poly_expansion(x, d):
    return np.vander(x, d, increasing=True)

def data_generator(n, f, sigma):
    x = np.random.uniform(-1.5, 1.5, n)
    y = np.random.normal(f(x), sigma**0.5, n)
    return x.tolist(), y.tolist()

def posterior(X, y, sigma, tau):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    I = np.eye(X.shape[1])
    covar = np.linalg.inv((X.T @ X) / sigma + I / tau)
    beta_hat = covar @ (X.T @ y) / sigma
    return beta_hat, covar

def posterior_predictive(X, beta_hat, covar, sigma):
    pred = X @ beta_hat
    std = np.sqrt(np.diag(X @ covar @ X.T) + sigma)
    return pred.flatten(), std.flatten()

def model_selection(x, y, sigma, tau):
    numbers = len(y)
    _min, _max = 1, 20
    scores = []
    
    for d in range(_min, _max + 1):
        try:
            X = poly_expansion(x, d)
            if X.shape[0] < X.shape[1]: 
                scores.append(-np.inf)
                continue
            beta_hat, covar = posterior(X, y, sigma, tau)
            y_pred = (X @ beta_hat).flatten()  
            log_likelihood = multivariate_normal.logpdf(y, mean=y_pred, cov=sigma * np.eye(numbers))
            bic = log_likelihood - 0.5 * d * np.log(numbers)
            scores.append(bic)
        except np.linalg.LinAlgError:
            scores.append(-np.inf)
    
    best_polynomial_degree = np.argmax(scores) + _min
    return best_polynomial_degree

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="Ex5: Bayesian Linear Regression", required=False)
    parser.add_argument("--n", type=int, default=30, required=False)
    parser.add_argument("--plot-boundaries", type=float, nargs=4, default=[-1.5, 1.5, -1.5, 3], required=False)
    parser.add_argument("--plot-resolution", type=int, default=100, required=False)
    parser.add_argument("--scatter-size", type=int, default=20, required=False)
    args = parser.parse_args()
    init()

