import torch
import torch.nn as nn

from load_hyperparameters import hp


# returns a modulelist used for a single layer of a UNET architecture
def double_conv_3d(in_features, out_features):
    return nn.ModuleList([
        nn.Conv3d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False),
        nn.BatchNorm3d(out_features),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=out_features,
                  out_channels=out_features,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm3d(out_features),
        nn.ReLU(inplace=True)
    ])


class UNet3D(nn.Module):
    def __init__(
            self,
            in_channels=len(hp["channel_names"]),  # number of channels: t1, t1ce, t2, flair
            out_channels=4,  # number of classes: background, edema, non-enhancing, GD-enhancing
            n_features=None,
            activation=nn.Sigmoid()):  # default for UNET
        super(UNet3D, self).__init__()

        # initialise member vars
        self.in_channels = in_channels
        self.out_channels = out_channels
        if n_features is None:
            self.n_features = [64, 128, 256, 512]
        else:
            self.n_features = n_features

        # generate layers
        self.contracting_layers = nn.ModuleList()
        self.expanding_layers = nn.ModuleList()

        # generate contracting layers
        self.contracting_layers.append(double_conv_3d(self.in_channels, self.n_features[0]))
        self.contracting_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

        for n in range(len(self.n_features) - 1):
            in_n = self.n_features[n]
            out_n = self.n_features[n + 1]
            self.contracting_layers.append(double_conv_3d(in_n, out_n))
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
            self.expanding_layers.append(double_conv_3d(n * 2, n))

        # compresses numerous feature channels down to one channel per class
        self.output_layer = nn.Conv3d(
            in_channels=list(reversed(self.n_features))[-1],  # same as n_features[0]
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1)

        self.activation = activation

    # x is passed in as [batch, input channels, dimensions]
    def forward(self, x):
        skipped_connections = []

        # run the contracting layers
        for layer in self.contracting_layers:
            if type(layer) == nn.ModuleList:
                for module in layer:
                    x = module(x)
                skipped_connections.append(x)
            else:
                x = layer(x)

        # run the bottleneck layer
        for module in self.bottleneck_layer:
            x = module(x)

        # run the extracting layers
        up_moves = 0
        for layer in self.expanding_layers:
            if type(layer) == nn.ModuleList:
                spatial_data = skipped_connections[-(up_moves + 1)]

                # if your input is of the wrong dimension,
                # it will need to be reshaped
                if x.shape != spatial_data.shape:
                    if spatial_data.shape[2] < x.shape[2]:
                        for i in range(2, len(spatial_data.shape)):
                            x = x.narrow(i, 0, spatial_data.shape[i])
                    if spatial_data.shape[2] > x.shape[2]:
                        for i in range(2, len(x.shape)):
                            spatial_data = spatial_data.narrow(i, 0, x.shape[i])

                # add spatial information
                x = torch.cat((spatial_data, x), dim=1)

                for module in layer:
                    x = module(x)
                up_moves += 1
            else:
                x = layer(x)

        # compress features down to number of output channels
        x = self.output_layer(x)

        return self.activation(x)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    test_model = UNet3D().to(DEVICE)
    # print("model architecture = "+ str(test_model))
    x = torch.rand(1, 4, 64, 64, 64).to(DEVICE)
    y = torch.rand(1, 4, 63, 63, 63).to(DEVICE)
    z = torch.rand(1, 4, 65, 65, 65).to(DEVICE)
    print("output shape x = " + str(test_model(x).shape))
    print("output shape y = " + str(test_model(y).shape))
    print("output shape z = " + str(test_model(z).shape))
