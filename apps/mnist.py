import sys

sys.path.append("../python/")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os


def MLPNet(dim, hidden_dim=200, num_classes=10, device = None):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim, device),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes, device),
    )
    return model


def epoch(dataloader, model, opt=None, device=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    hit, total = 0, 0
    loss_func = nn.SoftmaxLoss()
    loss_total = 0
    if opt is not None:
        model.train()
        for idx, data in enumerate(dataloader):
            x, y = data
            x._device = device
            y._device = device
            output = model(x)
            opt.reset_grad()
            loss = loss_func(output, y)
            loss_total += loss.numpy()
            loss.backward()
            opt.step()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    else:
        model.eval()
        for idx, data in enumerate(dataloader):
            x, y = data
            x._device = device
            y._device = device
            output = model(x)
            loss = loss_func(output, y)
            loss_total += loss.numpy()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    acc = (total - hit) / total
    return acc, loss_total / (idx + 1)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    hidden_dim=200,
    data_dir="data/mnist",
    device = ndl.cpu()
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_img_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_label_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_img_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_label_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    train_dataset = ndl.data.MNISTDataset(train_img_path, train_label_path)
    test_dataset = ndl.data.MNISTDataset(test_img_path, test_label_path)

    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = ndl.data.DataLoader(test_dataset)
    model = MLPNet(784, hidden_dim=hidden_dim, device = device)
    opt = optimizer(model.parameters(), lr=lr)

    for _ in range(epochs):
        train_err_rate, train_loss = epoch(train_dataloader, model, opt, device)
        print(
            "average error rate: %.2f, average training loss: %.2f"
            % (train_err_rate, train_loss)
        )
    test_err_rate, test_loss = epoch(test_dataloader, model)
    print("test error rate: %.2f" % test_err_rate)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    print(sys.path)
    train_mnist(data_dir="../data/mnist")
