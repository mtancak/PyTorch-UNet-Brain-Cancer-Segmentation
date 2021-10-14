import torch
import torch.nn as nn


class UNET(nn.Module):
    def __init__(
            self, 
            in_c = 4,  # number of channels: t1, t1ce, t2, flair
            out_c = 4,  # number of classes: background, edema, non-enhancing, GD-enhancing
            n_f = [64, 128, 256, 512]):  # default for UNET
        super(UNET, self).__init__()
        
        # initialise member vars
        self.in_channels = in_c
        self.out_channels = out_c
        self.n_features = n_f
        
        # generate layers
        self.contracting_layers = nn.ModuleList()
        self.bottleneck_layers = nn.ModuleList()
        self.expanding_layers = nn.ModuleList()
        
        # generate contracting layers
        for n in self.n_features:
            print("v " + str(n))
        
        # generate contracting layers
        for n in reversed(self.n_features):
            print("^ " + str(n))
        
        # compresses numerous feature channels down to one channel per class
        self.output_layer = nn.Conv3d(
            in_channels = self.n_features[0], 
            out_channels = self.out_channels, 
            kernel_size = 1, 
            stride = 1, 
            bias = False)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    test_model = UNET().to(device)