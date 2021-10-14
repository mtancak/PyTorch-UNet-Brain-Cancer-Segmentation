# a class that will take raw data directories and prepare
# the data for training
class DatasetCreator:
    def __init__(
            self,
            training_dir, 
            validation_dir, 
            channel_names, 
            segmentation_channel_name,
            min_data_dimension = 64):
        print("DatasetCreator.__init__()")

if __name__ == "__main__":
    dc = DatasetCreator(None, None, None, None, None)