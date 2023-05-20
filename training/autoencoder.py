import torch
from torch import nn


# convolutional LSTM autoencoder
class JatayuNet(nn.Module):
    def __init__(self, input_dimension=(1, 256, 256)):
        super().__init__()
        conv_seq1 = nn.Sequential(
            # input size 10 frame of 256 x 256 x 1 (Grayscale image)
            nn.Conv2d(in_channels=1, out_channels=128, stride=4, kernel_size=11),
            nn.LayerNorm([128, 256, 256]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, stride=2, kernel_size=5),
            nn.LayerNorm([64, 256, 256]),
        )
