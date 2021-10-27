import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from model import UNet3D
from brats20_dataset import BraTS20Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
NUMBER_OF_EPOCHS = 1000
SAVE_EVERY_X_EPOCHS = 15
SAVE_MODEL_LOC = "./model_"
LOAD_MODEL_LOC = None


# converts an array into a one hot vector. Source: https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
def onehot_initialization_v2(a, ncols=4):
    out = torch.zeros(a.numel(), ncols)
    out[torch.arange(a.numel()), a.ravel()] = 1
    return out.to(device=DEVICE)

# the f1 score in this paper uses a slightly modified way of computing the union which should result in a better f1 score for datasets with large class imbalance
# https://arxiv.org/pdf/1606.04797.pdf
def f1(probability, targets):
    probability = probability.flatten()
    targets = targets.flatten()
    assert (probability.shape == targets.shape)

    intersection = 2.0 * (probability * targets).sum()
    union = (probability * probability).sum() + (targets * targets).sum()
    dice_score = intersection / union
    return 1.0 - dice_score

# measures the f1 score of predictions at the end of an epoch
def f1_metric(model, loader):
    f1_score = 0.0
    with torch.no_grad():
        for data, seg in loader:
            pred = model(data)
            f1_score += f1(pred, onehot_initialization_v2(seg))

    f1_score /= len(loader)

    return f1_score.item()

# measures accuracy of predictions at the end of an epoch (bad for semantic segmentation)
def accuracy(model, loader, prin=False):
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for data, seg in loader:
            pred = model(data)
            pred = torch.argmax(pred, dim=1)

            correct_pixels += (pred == seg).sum()
            total_pixels += torch.numel(pred)

    return (correct_pixels / total_pixels).item()

# a training loop that runs a number of training epochs on a model
def train(model, loss_function, optimizer, train_loader, validation_loader):
    model_metric_scores = {}
    model_metric_scores["accuracy"] = []
    model_metric_scores["f1"] = []

    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        progress = tqdm(train_loader)
        
        for batch, (data, seg2) in enumerate(progress):
            pred = model(data)
            seg = onehot_initialization_v2(seg2)
            seg = torch.reshape(seg, (1, 64, 64, 64, 4))
            seg = torch.moveaxis(seg, 4, 1)
            print("pred shape = " + str(pred.shape))
            print("seg shape = " + str(seg.shape))
            loss = loss_function(pred, seg)

            # make the progress bar display loss
            progress.set_postfix(loss = loss.item())

            # back propagation
            optimizer.zero_grad()  # zeros out the gradients from previous batch
            loss.backward()
            optimizer.step()

        model.eval()

        model_metric_scores["accuracy"].append(accuracy(model, validation_loader))
        model_metric_scores["f1"].append(f1_metric(model, validation_loader))

        print("Accuracy for epoch (" + str(epoch) + ") is: " + str(model_metric_scores["accuracy"][-1]))
        print("F1 Score for epoch (" + str(epoch) + ") is: " + str(model_metric_scores["f1"][-1]))

        if (epoch % SAVE_EVERY_X_EPOCHS == 0 and SAVE_MODEL_LOC):
            torch.save(model.state_dict(), SAVE_MODEL_LOC + str(epoch))

        plt.scatter(range(0, epoch+1), model_metric_scores["accuracy"])
        plt.title("accuracy")
        plt.show()
        plt.scatter(range(0, epoch+1), model_metric_scores["f1"])
        plt.title("f1")
        plt.show()

    print("Final Accuracy for epoch is: " + str(accuracy(model, validation_loader)))
    print("Final DSC Score for epoch is: " + str(f1_metric(model, validation_loader)))


if __name__ == "__main__":
    # create/load model
    model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)

    if LOAD_MODEL_LOC != None:
        model.load_state_dict(torch.load(LOAD_MODEL_LOC))

    loss_function = f1
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = BraTS20Dataset(
        set_dir="C:/Users/Milan/Documents/Fast_Datasets/BraTS20/prep/train/",
        data_dir="data/",
        seg_dir="mask/")

    validation_dataset = BraTS20Dataset(
        set_dir="C:/Users/Milan/Documents/Fast_Datasets/BraTS20/prep/val/",
        data_dir="data/",
        seg_dir="mask/")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False)

    train(model, loss_function, optimizer, train_loader, validation_loader)