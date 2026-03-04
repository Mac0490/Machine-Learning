# Imports

import argparse
import numpy as np # type: ignore
import matplotlib # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torchvision # type: ignore
import torch.utils.data as torch_data # type: ignore
import torchvision.transforms as transforms # type: ignore

import sklearn # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# for MacOS:
# matplotlib.use('macosx')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt # type: ignore


# Neural Network


class NetSeq(nn.Module):
    def __init__(self):
        super(NetSeq, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#network = NetSeq()
#print(list(network.parameters()))

def train_nn(trainloader, testloader, lr, epochs):
    losses = []
    train_accs = []
    test_accs = []

    max_acc = None

    network = NetSeq()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    print("Start training")

    for epoch in range(epochs):

        print("Epoch: {}".format(epoch + 1))

        epoch_loss = []

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if i == 100:
                break

        losses.append(np.mean(epoch_loss))

        # Compute accuracy on training data

        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                inputs, labels = data
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on 60000 train images: %d %%"
            % (100 * correct / total)
        )
        train_accs.append(correct / total)

        # Validate all classes

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on 10000 test images: %d %%"
            % (100 * correct / total)
        )
        test_accs.append(correct / total)

    return network


# Support Vector Machine
from sklearn import svm # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

def train_svm(train_data, train_labels):
    model = svm.SVC(kernel='rbf', gamma='scale')  # You can try 'linear', 'poly', etc.
    model.fit(train_data, train_labels)

    return model


def test_svm(model, test_data, test_labels):
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)

    return accuracy


# Logistic Regression
from sklearn.linear_model import LogisticRegression # type: ignore

def train_log_reg(train_data, train_labels):
    model = LogisticRegression(max_iter=1000)
    model.fit(train_data, train_labels)

    return model


def test_log_reg(model, test_data, test_labels):
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)

    return accuracy


def init():

    lr = float(args.learning_rate)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch_data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    # Select mode from ['neural_network', 'svm', 'logistic_regression']

    # Note: you can either use the argparse library to pass the mode as an argument or set it manually

    mode = args.mode

    # mode = 'logistic_regression'
    # mode = 'svm'
    # mode = 'neural_network'

    if mode not in ["neural_network", "svm", "logistic_regression"]:
        raise ValueError(
            'Invalid mode, select from ["neural_network", "svm", "logistic_regression"]'
        )

    # TRAIN the model

    if mode == "neural_network":
        print("Training Neural Network")
        model = train_nn(trainloader, testloader, lr, epochs)

    elif mode in ["svm", "logistic_regression"]:
        train_data = trainset.data.numpy().reshape(-1, 28 * 28).astype("float32") / 255.0
        train_labels = trainset.targets.numpy()
        test_data = testset.data.numpy().reshape(-1, 28 * 28).astype("float32") / 255.0
        test_labels = testset.targets.numpy()

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        if mode == "svm":
            print("Training SVM Model")
            model = train_svm(train_data, train_labels)
            print("Evaluating SVM Model")
            svm_accuracy = test_svm(model, test_data, test_labels)
            print(f"SVM Accuracy: {svm_accuracy * 100:.2f} %")

        elif mode == "logistic_regression":
            print("Training Logistic Regression Model")
            model = train_log_reg(train_data, train_labels)
            print("Evaluating Logistic Regression Model")
            log_reg_accuracy = test_log_reg(model, test_data, test_labels)
            print(f"Logistic Regression Accuracy: {log_reg_accuracy * 100:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning-rate", default=0.001, required=False)
    parser.add_argument("--epochs", default=10, required=False)
    parser.add_argument("--batch-size", default=10, required=False)
    parser.add_argument("--mode", default="neural_network", required=False)

    args = parser.parse_args()
    init()


#Architecture Configuration: Input Layer- 784, Hidden Layers: 2 layers with 128 and 64 neurons, both using ReLU, Output Layer: 10 output neurons
#Why this architecture? simple feed-forward neural network (MLP), which is well-suited for MNIST, the two hidden layers are deep enough to capture relevant patterns but shallow enough to train quickly, ReLU activation accelerates convergence and avoids vanishing gradients
#Performance on Test Set: final test accuracy 92 percent, reach peak performance around epoch 10
#Trainig duration: ach epoch includes 100 batches, network converges in ~10 epochs, on a standard CPU, this typically takes a few minutes

#SVM - Data was flattened and scaled using StandardScaler, which improves SVM performance, Configuration- Used SVC with rbf kernel and default parameters, kernel='linear' – usually faster but lower accuracy, kernel='poly' – may improve fit but slower. Performance - full dataset, training an SVM is computationally expensive, VM scales poorly with sample size 60k samples, to speed it up we must reduce training set size to for example 10k samples, Typical accuracy: Around 94–97% with good parameter tuning and full data.

#Log. regression Configuration: Used LogisticRegression(max_iter=1000) from sklearn.linear_model
#Performance- achieves around 91–93% test accuracy on MNIST, fast to train and easy to interpret, less expressive than SVM or NN

#Overall Observations
#NN, accuracy 92 percents, Speed is medium fast, pros-learns nonlinearity, tunable, flexible, cons-needs tuning, sensitive to architecture and overfitting
#SVM -BRF kernel, 94-97 percents, Speed is Slow, Pros- powerful with small data, good generalization, Cons - does not scale well with big datasets
#Log regression, accuracy is 91-93 percents, speed is fast, Pros- Simple, interpretable, efficient, Cons - linear model, limited on complex data