# Imports

import argparse
import random
import numpy as np # type: ignore
import matplotlib # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt # type: ignore


class Net(nn.Module):
    def __init__(self, input_dim, hidden_0_dim, hidden_1_dim, hidden_2_dim, output_dim):
        super(Net, self).__init__()

        self.input = nn.Linear(input_dim, hidden_0_dim)
        self.hidden_1 = nn.Linear(hidden_0_dim, hidden_1_dim)
        self.hidden_2 = nn.Linear(hidden_1_dim, hidden_2_dim)
        self.output = nn.Linear(hidden_2_dim, output_dim)

    def forward(self, x):

        # TODO: Forward the signal x. Use ReLu activation. In the output player, use a linear activation.
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))        
        y_hat = self.output(x)

        return y_hat


def init():

    # Variables

    n = int(args.n)
    res = args.plot_resolution
    learningrate = float(args.learning_rate)
    epochs = int(args.epochs)

    # Generate Data

    x, y = data_generator(n, second_order_function)

    # Train model parameters

    model = train_nn(x, y, learningrate, epochs)

    # Predict

    px = [min(x) + i * (max(x) - min(x)) / res for i in range(res)]
    pxx = [[elem] for elem in px]
    inputs = torch.from_numpy(np.array(pxx, dtype="float32"))

    predictions = model(inputs)
    py = [elem.item() for elem in predictions]

    # Plot

    plot([x, y], [px, py])


def train_nn(x, y, lr, epochs):
    """Train neural network model"""

    # Pre-process data

    xx = [[elem] for elem in x]
    targets = [[elem] for elem in y]

    inputs = torch.from_numpy(np.array(xx, dtype="float32"))
    targets = torch.from_numpy(np.array(targets, dtype="float32"))

    # Initialize model

    network = Net(1, 64, 32, 16, 1)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # Train model parameters

    for i in range(epochs):
        optimizer.zero_grad()  # Reset the gradients

        outputs = network(inputs)

        loss = loss_function(outputs, targets)

        loss.backward()

        optimizer.step()

        print("MSE: {}".format(loss.item()))

    return network


def second_order_function(x):

    return x**2 - x


def data_generator(n, f):
    """Generate training data"""

    x = [3 * (random.random() - 0.5) for _ in range(n)]
    y = [f(x[i]) + 1 * (random.random() - 0.5) for i in range(len(x))]

    return x, y


def plot(observations, poly_estimate):
    """Plot data and regression"""

    plt.rc("xtick")
    plt.rc("ytick")
    plt.figure(figsize=(4.8, 4))
    plt.title(args.title)
    plt.axis(args.plot_boundaries)
    plt.plot(poly_estimate[0], poly_estimate[1], color="red")
    plt.scatter(observations[0], observations[1], s=args.scatter_size)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=True)
    plt.interactive(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments

    parser.add_argument("--title", default="Ex6.2: Neural Networks", required=False)

    parser.add_argument("--n", default=100, required=False)

    parser.add_argument("--learning-rate", default=0.001, required=False)

    parser.add_argument("--epochs", default=500, required=False)

    parser.add_argument(
        "--plot-boundaries",
        default=[-1.5, 1.5, -1.5, 3],  # min_x, max_x, min_y, max_y
        required=False,
    )

    parser.add_argument("--plot-resolution", default=100, required=False)

    parser.add_argument("--scatter-size", default=20, required=False)

    parser.add_argument("--font-size", default=10, required=False)

    args = parser.parse_args()

    init()
