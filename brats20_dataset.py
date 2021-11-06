import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt


class BraTS20Dataset(Dataset):
    def __init__(self, set_dir, data_dir, seg_dir):
        self.set_dir = set_dir
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.list_samples = os.listdir(self.set_dir + self.data_dir)
        check_list_samples = os.listdir(self.set_dir + self.seg_dir)
        if self.list_samples != check_list_samples:
            raise OSError("the data and segmentation directories have " + 
                          "different file names!")
    
    def __len__(self):
        return len(self.list_samples)

    # generate the file paths for data+mask, then load them in, move them to the right device and return them
    def __getitem__(self, index):
        data_img_path = self.set_dir + self.data_dir + self.list_samples[index]
        seg_img_path = self.set_dir + self.seg_dir + self.list_samples[index]
        image = np.load(data_img_path)
        mask = np.load(seg_img_path)
        image = torch.tensor(image, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask, dtype=torch.int64, device=self.device)
        return image, mask


if __name__ == "__main__":
    ds = BraTS20Dataset(
        set_dir="C:/Users/Milan/Documents/Fast_Datasets/BraTS20/prep/train/",
        data_dir="data/",
        seg_dir="mask/")
    plt.imshow(ds[0][0][0][10].cpu().numpy())
    plt.show()
    plt.imshow(ds[0][0][1][10].cpu().numpy())
    plt.show()
    plt.imshow(ds[0][0][2][10].cpu().numpy())
    plt.show()
    plt.imshow(ds[0][0][3][10].cpu().numpy())
    plt.show()
    plt.imshow(ds[0][1][10].cpu().numpy())
    plt.show()