# Imports

import argparse
import numpy as np # type: ignore
import matplotlib # type: ignore
import math
import random
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

import sklearn # type: ignore


# for MacOS:
# matplotlib.use('macosx')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt # type: ignore


# Neural Network


class NetSeq(nn.Module):
    def __init__(self):
        super(NetSeq, self).__init__()

        self.fc1 = nn.Linear(1, 64)      
        self.fc2 = nn.Linear(64, 64)      
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generative_function(x):
    return 6 * x**4 - 2 * x**3 - 3 * x**2 + 4 * x


def generate_data(n):
    x = [2 * (random.random() - 0.5) for _ in range(n)]
    y = [generative_function(x[i]) + 1 * (random.random() - 0.5) for i in range(len(x))]

    return x, y


def train_nn(x, y, lr, epochs):
    """Train neural network model"""
    # Pre-process data

    xx = [[elem] for elem in x]
    targets = [[elem] for elem in y]

    inputs = torch.from_numpy(np.array(xx, dtype="float32"))
    targets = torch.from_numpy(np.array(targets, dtype="float32"))

    network = NetSeq()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # Train model parameters

    for i in range(epochs):
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = loss_fn(targets, outputs)  # Compute prediction loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        print("MSE: {}".format(loss.item()))

    return network


def plot(observations, poly_estimate, title, bounds):
    """Plot data and regression"""

    plt.rc("xtick")
    plt.rc("ytick")
    plt.figure(figsize=(4.8, 4))
    plt.title(f"Mode: {title}")
    plt.axis(bounds)
    plt.plot(poly_estimate[0], poly_estimate[1], color="red")

    # plot the true generative function
    x = np.linspace(-1.5, 1.5, 100)
    y = [generative_function(xi) for xi in x]
    plt.plot(x, y, color="green")

    plt.scatter(observations[0], observations[1], s=20)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=True)
    plt.interactive(False)


def init():
    n = int(args.n)
    lr = float(args.learning_rate)
    epochs = int(args.epochs)

    mode = args.mode

    x, y = generate_data(n)
    px = [min(x) + i * (max(x) - min(x)) / 100 for i in range(100)]

    if mode == "neural_network":

        model = train_nn(x, y, lr, epochs)
        with torch.no_grad():
            px_tensor = torch.from_numpy(np.array([[xi] for xi in px], dtype="float32"))
            py_tensor = model(px_tensor)
            py = py_tensor.numpy().flatten().tolist()

    elif mode == "svm":

        from sklearn.svm import SVR  # type: ignore
        model = SVR(kernel="rbf", C=100, gamma="scale")
        model.fit(np.array(x).reshape(-1, 1), y)
        py = model.predict(np.array(px).reshape(-1, 1)).tolist()

    else:
        # in case of invalid mode
        py = [0 for _ in px]

    plot([x, y], [px, py], mode, [min(x), max(x), min(y), max(y)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument("--learning-rate", default=0.001, required=False)

    parser.add_argument("--epochs", default=100, required=False)

    parser.add_argument("--n", default=200, required=False)

    parser.add_argument("--mode", default="neural_network", required=False)

    args = parser.parse_args()

    init()

#DOCUMENTATION
#What does your architecture look like
#Input Layer: 1 input neuron (1D input feature), Hidden Layers- Two fully connected layers with 64 neurons each, Activation Function- ReLU, Output Layer- 1 output neuron for the regression target, Loss Function- Mean Squared Error (MSE), Optimizer- Adam with learning rate 0.001, Epochs- 100, Training Samples- 200 data points generated using the provided polynomial function + nois

# Why did you build your network like this
#The network is kept small and simple (2 hidden layers) for fast convergence and to avoid overfitting with limited data.Relu helps to indroduce non-linearity and stabilize training. 64 neurons provide sufficient capacity to model the nonlinear generative function without excessive computational cost.

#Settings-SVM
#Library: sklearn.svm.SVR, RBF kernel usually performs well with nonlinear functions. Polynomial Degree 3. Paramenters: C=100, gamma='scale', Training is fast on small datasets but does not scale well with large datasets or high-dimensional data.

#What are your general findings? Are there advantages/disadvantages to the methods? 
#NN- Pross
#NNs can approximate highly nonlinear functions when provided with sufficient depth, width, and training. There is a noise Tolerance, and it producesa continues and differentiable output, suitable for modeling smoth generative process.
#NN- Cons
#Performance depends on architecture (number of layers), activation functions, and learning rate. May underfit with very small datasets. Sensitive to initialization and hyperparameters; might get stuck in local minima.

#SVM- Pross
#Good performance in small sample settings, especially with the RBF kernel.SVMs can perform well even when feature space is expanded.

#SVM -Cons
#SVMs scale poorly with large datasets, May require feature engineering or kernel selection for good performance.


# Are there settings that don’t work at all?
# Linear kernel SVM fails to capture nonlinearity, too shallow neural networks (no hidden layers or very small hidden size) result in poor function approximation.