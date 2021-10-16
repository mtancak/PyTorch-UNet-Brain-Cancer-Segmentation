import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt

class BraTS20Dataset(Dataset):
    def __init__(self, image_dir, data_name, mask_name, extension):
        self.image_dir = image_dir
        self.data_name = data_name
        self.mask_name = mask_name
        self.extension = extension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.list_of_images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.list_of_images)
    
    def __getitem__(self, index):
        img_path = self.image_dir + self.list_of_images[index]
        image = np.load(img_path)
        mask = np.load(img_path)
        image = torch.tensor(image, dtype=torch.float64, device=self.device)
        mask = torch.tensor(mask, dtype=torch.float64, device=self.device)
        return image, mask


if __name__ == "__main__":
    ds = BraTS20Dataset(
        image_dir="C:/Users/Milan/Documents/Fast_Datasets/BraTS20/prep/train/",
        data_name="_data",
        mask_name="_mask",
        extension=".npy")
    plt.imshow(ds[0][0][0][10].cpu().numpy())
    plt.show()