import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import UNet3D
from brats20_dataset import BraTS20Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
NUMBER_OF_EPOCHS = 5


# measures accuracy of a model at the end of an epoch (bad for semantic segmentation)
def accuracy(model, loader):
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for data, mask in loader:
            pred = model(data)
            pred = torch.argmax(pred, dim=1)
            correct_pixels += (pred == mask).sum()
            total_pixels += torch.numel(pred)

    return (correct_pixels / total_pixels).item()

# a training loop that runs a number of training epochs on a model
def train(model, loss_function, optimizer, train_loader, validation_loader):
    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        progress = tqdm(train_loader)
        
        for batch, (data, seg), in enumerate(progress):
            pred = model(data)
            loss = loss_function(pred, seg)

            # make the progress bar display loss
            progress.set_postfix(loss = loss.item())

            # back propagation
            optimizer.zero_grad()  # zeros out the gradients from previous batch
            loss.backward()
            optimizer.step()

        model.eval()
        print("Accuracy for epoch (" + str(epoch) + ") is: " + str(accuracy(model, validation_loader)))

if __name__ == "__main__":
    # create model
    model = UNet3D(in_channels = 4, out_channels = 4).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    train_dataset = BraTS20Dataset(
        set_dir = "C:/Users/Milan/Documents/Fast_Datasets/BraTS20/prep/train/",
        data_dir = "data/",
        seg_dir = "mask/")
    
    validation_dataset = BraTS20Dataset(
        set_dir = "C:/Users/Milan/Documents/Fast_Datasets/BraTS20/prep/val/",
        data_dir = "data/",
        seg_dir = "mask/")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = 0,
        shuffle = True)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size = 1,
        num_workers = 0,
        shuffle = False)

    train(model, loss_function, optimizer, train_loader, validation_loader)