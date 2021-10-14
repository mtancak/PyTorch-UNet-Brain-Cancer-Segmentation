import torch
import torch.nn as nn


class UNET(nn.Module):
    def __init__(
            self, 
            in_channels = 4,  # number of channels: t1, t1ce, t2, flair
            out_channels = 4,  # number of classes: background, edema, non-enhancing, GD-enhancing
            layer_features = [64, 128, 256, 512]):  # default for UNET
        super(UNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_features = layer_features


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    test_model = UNET().to(device)