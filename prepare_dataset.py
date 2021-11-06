import os
import numpy as np
from sklearn.model_selection import train_test_split
import nibabel as nib  # reading .nii files
from tqdm import tqdm  # progress bar
import patchify  # splits dataset into patches

SAVE = True


# a class that will take raw data directories and prepare
# the data for training
class DatasetCreator:
    def __init__(self, hp_fn=None):
        self.input_data_dir = ""
        self.output_training_dir = ""
        self.output_validation_dir = ""
        self.min_patch_size = 64
        self.do_bg_threshold = False
        self.channel_names = []
        self.seg_channel_name = "seg"
        self.sample_names = []
        self.save_thresholded_sample_name = False
        self.data_dir_n = ""
        self.seg_dir_n = ""

        if isinstance(hp_fn, str):
            with open(hp_fn, "r") as hp:
                entries = hp.readlines()
                for entry in entries:
                    p, v = entry.split(":", 1)
                    v = v.rstrip()  # removes trailing space
                    if p == "input_data_dir":
                        self.input_data_dir = v
                    if p == "output_training_dir":
                        self.output_training_dir = v
                    if p == "output_validation_dir":
                        self.output_validation_dir = v
                    if p == "do_bg_threshold":
                        self.do_bg_threshold = v
                    if p == "seg_channel_name":
                        self.seg_channel_name = v
                    if p == "save_thresholded_sample_name":
                        self.save_thresholded_sample_name = v == "True"
                    if p == "min_patch_size":
                        self.min_patch_size = int(v)
                    if p == "channel_names":
                        self.channel_names = v.split(",")
                    if p == "data_dir_n":
                        self.data_dir_n = v
                    if p == "seg_dir_n":
                        self.seg_dir_n = v

        if not os.path.isdir(self.output_training_dir + self.data_dir_n):
            os.mkdir(self.output_training_dir + self.data_dir_n)
        if not os.path.isdir(self.output_training_dir + self.seg_dir_n):
            os.mkdir(self.output_training_dir + self.seg_dir_n)
        if not os.path.isdir(self.output_validation_dir + self.data_dir_n):
            os.mkdir(self.output_validation_dir + self.data_dir_n)
        if not os.path.isdir(self.output_validation_dir + self.seg_dir_n):
            os.mkdir(self.output_validation_dir + self.seg_dir_n)

    # remove images where there is too little non-bg data
    def bg_threshold(self):
        if self.do_bg_threshold == "No":
            return

        if self.do_bg_threshold == "Load":
            self.sample_names = np.load("sample_names.npy")
        elif self.do_bg_threshold == "Compute":
            for sample_name in self.sample_names:
                seg_data = nib.load(
                    self.input_data_dir + sample_name + "/" +
                    sample_name + "_" + self.seg_channel_name
                    + ".nii").get_fdata()
                uniques = np.unique(seg_data, return_counts=True)[1]
                uniques = uniques / np.sum(uniques)

                if np.max(uniques) > 0.99:
                    self.sample_names.remove(sample_name)

            if self.save_thresholded_sample_name:
                np.save("sample_names.npy", np.array(self.sample_names))

    # check which files already exist in the training 
    # and validation directories to ignore repeats when saving data
    def filter_already_saved_sampled(self, train_samples, val_samples):
        for entry in (os.listdir(self.output_training_dir) +
                      os.listdir(self.output_validation_dir)):
            entry = entry.split("__")[0]
            if entry in train_samples:
                train_samples.remove(entry)
            elif entry in val_samples:
                val_samples.remove(entry)

    # load the appropriate .nii files and convert to numpy arrays
    def load_data(self, sample):
        seg_data = nib.load(self.input_data_dir + sample + "/" +
                            sample + "_" + self.seg_channel_name + ".nii").get_fdata()
        seg_data[seg_data == 4] = 3  # 3rd class is labelled as 4

        patch_channels = []
        for channel in self.channel_names:
            patch_channels.append(nib.load(self.input_data_dir +
                                           sample + "/" + sample + "_" + channel + ".nii").get_fdata())

        data = np.stack(np.array(patch_channels), axis=0)

        return seg_data, data

    # save data in target output directories
    def save(self):
        train_samples, val_samples = train_test_split(
            self.sample_names,
            test_size=0.3,
            random_state=42)

        self.filter_already_saved_sampled(train_samples, val_samples)

        # go through each data sample that needs to be saved
        # and generate patches
        for data_type in ["training", "validation"]:
            if data_type == "training":
                samples = train_samples
                loc = self.output_training_dir
            else:
                samples = val_samples
                loc = self.output_validation_dir
            for sample in (tqdm(samples)):
                seg_data, data = self.load_data(sample)

                if seg_data is None:
                    continue

                # use patchify to split data into patches
                data = patchify.patchify(data, (4, self.min_patch_size, self.min_patch_size, self.min_patch_size),
                                         step=self.min_patch_size).squeeze(0)  # squeeze 0 to account for channel dim
                seg_data = patchify.patchify(seg_data, (self.min_patch_size, self.min_patch_size, self.min_patch_size),
                                             step=self.min_patch_size)

                # save patches as separate numpy files
                if SAVE:
                    for z in range(seg_data.shape[0]):
                        for y in range(seg_data.shape[1]):
                            for x in range(seg_data.shape[2]):
                                np.save(loc + self.data_dir_n + sample + "_"
                                        + str(z) + "_"
                                        + str(y) + "_"
                                        + str(x) + ".npy",
                                        data[z][y][x])
                                np.save(loc + self.seg_dir_n + sample + "_"
                                        + str(z) + "_"
                                        + str(y) + "_"
                                        + str(x) + ".npy",
                                        seg_data[z][y][x])

    # run the dataset generation with the () operator
    def __call__(self):
        self.sample_names = os.listdir(self.input_data_dir)

        self.bg_threshold()
        self.save()


if __name__ == "__main__":
    dc = DatasetCreator("hyperparameters.txt")
    dc()
