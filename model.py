import torch
import torch.nn as nn

# returns a modulelist used for a single layer of a UNET architecture
def double_conv_3d(in_f, out_f):
    return nn.ModuleList([
        nn.Conv3d(
            in_channels=in_f, 
            out_channels=out_f,
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False),
        nn.BatchNorm3d(out_f),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=out_f, 
                  out_channels=out_f, 
                  kernel_size=3, 
                  stride=1, 
                  padding=1, 
                  bias=False),
        nn.BatchNorm3d(out_f),
        nn.ReLU(inplace=True)
    ])

class UNET3D(nn.Module):
    def __init__(
            self, 
            in_c = 4,  # number of channels: t1, t1ce, t2, flair
            out_c = 4,  # number of classes: background, edema, non-enhancing, GD-enhancing
            n_f = [64, 128, 256, 512]):  # default for UNET
        super(UNET3D, self).__init__()
        
        # initialise member vars
        self.in_channels = in_c
        self.out_channels = out_c
        self.n_features = n_f
        
        # generate layers
        self.contracting_layers = nn.ModuleList()
        self.bottleneck_layers = nn.ModuleList()
        self.expanding_layers = nn.ModuleList()
        
        # generate contracting layers
        self.contracting_layers += double_conv_3d(self.in_channels, self.n_features[0])
        self.contracting_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        
        for n in range(len(self.n_features) - 1):
            in_n = self.n_features[n]
            out_n = self.n_features[n + 1]
            self.contracting_layers += double_conv_3d(in_n, out_n)
            self.contracting_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        
        # generating the bottleneck layer
        self.bottleneck_layer = double_conv_3d(self.n_features[-1], self.n_features[-1] * 2)
        
        # generate expanding layers
        for n in reversed(self.n_features):
            self.expanding_layers.append(
                nn.ConvTranspose3d(
                    n * 2, 
                    n, 
                    kernel_size=2, 
                    stride=2))
            self.expanding_layers += double_conv_3d(n * 2, n)
        
        # compresses numerous feature channels down to one channel per class
        self.output_layer = nn.Conv3d(
            in_channels = list(reversed(self.n_features))[-1],  # same as n_features[0]
            out_channels = self.out_channels, 
            kernel_size = 1, 
            stride = 1, 
            bias = False)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    test_model = UNET3D().to(device)
    print(test_model)