# Imports
#Remove the Max Pooling layers
import argparse
import numpy as np # type: ignore
import matplotlib # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torchvision # type: ignore
import torch.utils.data as torch_data # type: ignore
import torchvision.transforms as transforms # type: ignore

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt # type: ignore


class NetSeq(nn.Module):
    def __init__(self):
        super(NetSeq, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
    
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            
            nn.Dropout(0.2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 24 * 24, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)

        return x


def imshow(img):
    """Plot images"""

    plt.title(args.title)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices


def init():

    # Variables

    classes = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    )
    lr = float(args.learning_rate)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)

    # Load data

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

    # Train model or load a trained model

    if not args.model_path:
        network = train_nn(trainloader, testloader, classes, batch_size, lr, epochs)
    else:
        network = NetSeq()
        
        network.load_state_dict(torch.load(args.model_path))
        
        network.eval()

    classify(network, testset, classes)


def train_nn(trainloader, testloader, classes, batch_size, lr, epochs):
    """Train neural network model"""

    network = NetSeq()

    # TODO: Initialize cross entropy loss function

    criterion = nn.NLLLoss()

    # TODO: Initialize Adam optimizer

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    train_losses, train_accs, test_accs = [], [], []

    print("Start training")

    for epoch in range(epochs):
        network.train()
        running_loss = 0.0

        print("Epoch: {}".format(epoch + 1))

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate all class
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        network.eval()

        train_correct, train_total = 0, 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                outputs = network(images)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        train_acc = train_correct / train_total
        train_accs.append(train_acc)

        test_correct, test_total = 0, 0
        correct = [0]*10
        total = [0]*10
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = network(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                for label, pred in zip(labels, predicted):
                    if label == pred:
                        correct[label] += 1
                    total[label] += 1

        test_acc = test_correct / test_total
        test_accs.append(test_acc)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        for i in range(10):
            print(f'Class {i} Acc: {correct[i]/total[i]:.4f}')

        # Save model
        torch.save(network.state_dict(), f'mnist_epoch_{epoch+1}.pth')

    # Plotting
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    torch.save(
            network.state_dict(),
            f"./mnist_epoch_{epoch + 1}_acc_{test_acc:.4f}.pth",
        )

    return network


def classify(network, data, classes):
    """Classify input"""

    with torch.no_grad():
        stack_tot = torch.tensor([])
        for i in range(0, 10):
            stack_label = torch.tensor([])
            stack_predicted = torch.tensor([], dtype=torch.int64)
            idx = get_indices(data, i)
            dataloader = torch.utils.data.DataLoader(
                data,
                batch_size=1,
                shuffle=False,
                sampler=torch_data.SubsetRandomSampler(idx),
            )

            for data_item in dataloader:
                inputs, labels = data_item
                # print(inputs.shape)
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)

                if predicted.item() != i:
                    stack_label = torch.cat((stack_label, inputs), 0)
                    stack_predicted = torch.cat((stack_predicted, predicted), 0)

                if len(stack_label) == 10:
                    stack_tot = torch.cat((stack_tot, stack_label), 0)
                    print(
                        "False classifications for class {}: ".format(i)
                        + " ".join(
                            classes[stack_predicted[j]]
                            for j in range(len(stack_predicted))
                        )
                    )
                    break

        imshow(torchvision.utils.make_grid(stack_tot, nrow=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments

    parser.add_argument(
        "--title", default="False Classifications for each Class", required=False
    )

    parser.add_argument("--learning-rate", default=0.001, required=False)

    parser.add_argument("--epochs", default=4, required=False)

    parser.add_argument("--model-path", default="", required=False)

    parser.add_argument("--batch-size", default=10, required=False)

    parser.add_argument("--font-size", default=10, required=False)

    args = parser.parse_args()

    init()

#Result effect on the training result: Low and stable loss, high accuracy: Train-slight overfitting, Test-strong generalization
#For MNIST, these results are near state-of-the-art. Further improvements would require architectural changes, not just hyperparameter tuning.
# What does a Max Pooling layer do? - A Max Pooling layer downsamples feature maps by outputting the maximum value in sliding windows (e.g., 2×2). Reduces size (speeds up computation), Preserves key features (ignores noise), Improves translation invariance (e.g., for MNIST digits), removing it often lowers accuracy and increases errors, as seen in your results. Use it when spatial position matters less than overall patterns.