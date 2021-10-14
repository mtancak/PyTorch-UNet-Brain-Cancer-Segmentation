import os
import numpy as np
import nibabel as nib

# a class that will take raw data directories and prepare
# the data for training
class DatasetCreator:
    def __init__(self, hp_fn=None):
        self.input_data_dir = ""
        self.output_training_dir = ""
        self.output_validation_dir = ""
        self.do_bg_threshold = False
        self.seg_channel_name = "seg"
        self.sample_names = []
        
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
                        self.do_bg_threshold = v == "True"
                    if p == "seg_channel_name":
                        self.seg_channel_name = v
                        
    # remove images where there is too little non-bg data
    def bg_threshold(self):
        if not self.do_bg_threshold:
            return
        for sample_name in self.sample_names:
            seg_img = nib.load(
                self.input_data_dir + sample_name + "/" + 
                sample_name + "_" + self.seg_channel_name 
                + ".nii").get_fdata()
            uniques = np.unique(seg_img, return_counts=True)[1]
            print("uniques = " + str(uniques))
            uniques = uniques / np.sum(uniques)
        
            if np.max(uniques) > 0.99:
                self.sample_names.remove(sample_name)

    def __call__(self):
        self.sample_names = os.listdir(self.input_data_dir)
        
        self.bg_threshold()

if __name__ == "__main__":
    dc = DatasetCreator("hyperparameters.txt")
    dc()