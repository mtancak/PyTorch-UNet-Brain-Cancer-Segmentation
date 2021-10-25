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
NUMBER_OF_EPOCHS = 100

def onehot_initialization_v2(a, ncols=4):
    out = torch.zeros(a.numel(), ncols)
    out[torch.arange(a.numel()), a.ravel()] = 1
    return out.to(device=DEVICE)

def f1(probability, targets):
    probability = nn.Sigmoid()(probability.flatten())
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

    return f1_score

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
    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        progress = tqdm(train_loader)
        
        for batch, (data, seg2), in enumerate(progress):
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
        print("Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy(model, validation_loader)))
        print("F1 Score for epoch (" + str(epoch) + ") is: " + str(f1_metric(model, validation_loader)))
        plt.imshow(data[0][0][20].detach().cpu().numpy())
        plt.show()
        plt.imshow(torch.argmax(pred, dim=1)[0][20].detach().cpu().numpy())
        plt.show()
        plt.imshow(seg2[0][20].detach().cpu().numpy())
        plt.show()
        plt.imshow(data[0][0][20].detach().cpu().numpy())
        plt.show()
        plt.imshow(torch.argmax(pred, dim=1)[0][20].detach().cpu().numpy())
        plt.show()
        plt.imshow(seg2[0][20].detach().cpu().numpy())
        plt.show()
    print("Final Accuracy for epoch is: " + str(accuracy(model, validation_loader)))
    print("Final DSC Score for epoch is: " + str(f1_metric(model, validation_loader)))
    print("Print: ")
    # print_predictions(model, validation_loader)


if __name__ == "__main__":
    # create model
    model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)

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