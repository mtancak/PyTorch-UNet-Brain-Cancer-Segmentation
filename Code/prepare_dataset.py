import os
import numpy as np
from sklearn.model_selection import train_test_split
import nibabel as nib  # reading .nii files
from tqdm import tqdm  # progress bar
import patchify  # splits dataset into patches

from load_hyperparameters import hp

SAVE = True


# a class that will take raw data directories and prepare
# the data for training
class DatasetCreator:
    def __init__(self, hp_fn=None):
        self.sample_names = []

    # remove samples where there is too little non-bg data
    def preprocess(self):
        for sample_name in self.sample_names:
            seg_data = nib.load(
                hp["raw_data_dir"] + sample_name + "/" +
                sample_name + "_" + hp["seg_channel_name"]
                + ".nii").get_fdata()
            uniques = np.unique(seg_data, return_counts=True)[1]
            uniques = uniques / np.sum(uniques)

            if np.max(uniques) > hp["bg_threshold"]:
                self.sample_names.remove(sample_name)

        if hp["save_thresholded_sample_name"]:
            np.save("sample_names.npy", np.array(self.sample_names))

    # check which files already exist in the training 
    # and validation directories to ignore repeats when saving data
    def filter_already_saved_sampled(self, train_samples, val_samples):
        for entry in (os.listdir(hp["training_dir"]) +
                      os.listdir(hp["validation_dir"])):
            entry = entry.split("__")[0]
            if entry in train_samples:
                train_samples.remove(entry)
            elif entry in val_samples:
                val_samples.remove(entry)

    # load the appropriate .nii files and convert to numpy arrays
    def load_data(self, sample):
        seg_data = nib.load(hp["raw_data_dir"] + sample + "/" +
                            sample + "_" + hp["seg_channel_name"] + ".nii").get_fdata()
        seg_data[seg_data == 4] = 3  # 3rd class is labelled as 4

        patch_channels = []
        for channel in hp["channel_names"]:
            patch_channels.append(nib.load(hp["raw_data_dir"] +
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
                loc = hp["training_dir"]
            else:
                samples = val_samples
                loc = hp["validation_dir"]
                
            for sample in (tqdm(samples)):
                seg_data, data = self.load_data(sample)

                if seg_data is None:
                    continue

                # use patchify to split data into patches
                data = patchify.patchify(data, (len(hp["channel_names"]), hp["min_patch_size"], hp["min_patch_size"], hp["min_patch_size"]),
                                         step=hp["min_patch_size"]).squeeze(0)  # squeeze 0 to account for channel dim
                seg_data = patchify.patchify(seg_data, (hp["min_patch_size"], hp["min_patch_size"], hp["min_patch_size"]),
                                             step=hp["min_patch_size"])

                # save patches as separate numpy files
                if SAVE:
                    for z in range(seg_data.shape[0]):
                        for y in range(seg_data.shape[1]):
                            for x in range(seg_data.shape[2]):
                                np.save(loc + hp["data_dir_name"] + sample + "_"
                                        + str(z) + "_"
                                        + str(y) + "_"
                                        + str(x) + ".npy",
                                        data[z][y][x])
                                np.save(loc + hp["seg_dir_name"] + sample + "_"
                                        + str(z) + "_"
                                        + str(y) + "_"
                                        + str(x) + ".npy",
                                        seg_data[z][y][x])

    # run the dataset generation with the () operator
    def __call__(self):


        if hp["load_data"]:
            self.sample_names = np.load("sample_names.npy")
        else:
            self.sample_names = os.listdir(hp["raw_data_dir"])

        if hp["do_preprocess"] == "Compute":
            self.preprocess()
        self.save()


if __name__ == "__main__":
    dc = DatasetCreator("hyperparameters.txt")
    dc()
