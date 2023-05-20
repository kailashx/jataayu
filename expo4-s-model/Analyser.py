import torch
from torch import nn
from torch.utils.benchmark import timer


class Analyser(nn.Module):
    def __init__(self):
        super(Analyser, self).__init__()
        self.conv_enc = nn.Sequential(
            # Batch, Channel, Height, Width, Depth

            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=4, padding=2),  # layer-1
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=4, padding=2),  # layer-1
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool2d(100),

            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=4, padding=2),  # layer-2
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=15, out_channels=20, kernel_size=4, padding=2),  # layer-2
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool2d(80),

            nn.Conv2d(in_channels=20, out_channels=25, kernel_size=4, padding=2),  # layer-3
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=25, out_channels=30, kernel_size=4, padding=2),  # layer-3
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool2d(60),

            nn.Conv2d(in_channels=30, out_channels=35, kernel_size=4, padding=2),  # layer-4
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=35, out_channels=40, kernel_size=4, padding=2),  # layer-4
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool2d(40),

            nn.Conv2d(in_channels=40, out_channels=45, kernel_size=4, padding=2),  # layer-5
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=45, out_channels=50, kernel_size=4, padding=2),  # layer-5
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool2d(16),

            nn.Flatten(start_dim=2, end_dim=3),

        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.SELU(inplace=True),
            nn.Linear(128, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 90),
            nn.SELU(inplace=True),
            nn.Linear(90, 80),
            nn.SELU(inplace=True),
            nn.Linear(80, 90),
            nn.SELU(inplace=True),
            nn.Linear(90, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 128),
            nn.SELU(inplace=True),
            nn.Linear(128, 256),
            nn.SELU(inplace=True),
        )
        self.conv_dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=50, out_channels=45, kernel_size=4),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(in_channels=45, out_channels=40, kernel_size=4),
            nn.SELU(inplace=True),
            nn.UpsamplingBilinear2d(40),
            # Debug(),
            nn.ConvTranspose2d(in_channels=40, out_channels=35, kernel_size=4),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(in_channels=35, out_channels=30, kernel_size=4),
            nn.SELU(inplace=True),
            nn.UpsamplingBilinear2d(60),
            # Debug(),
            nn.ConvTranspose2d(in_channels=30, out_channels=25, kernel_size=4),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(in_channels=25, out_channels=20, kernel_size=4),
            nn.SELU(inplace=True),
            nn.UpsamplingBilinear2d(80),
            # Debug(),
            nn.ConvTranspose2d(in_channels=20, out_channels=15, kernel_size=4),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(in_channels=15, out_channels=10, kernel_size=4),
            nn.SELU(inplace=True),
            nn.UpsamplingBilinear2d(100),
            # Debug(),
            nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=4),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=4),
            nn.SELU(inplace=True),
            nn.UpsamplingBilinear2d(120),
            # Debug(),
        )

    def forward(self, input_footage):
        input_footage = self.conv_enc(input_footage)
        input_footage = self.fc(input_footage)
        # Reshaping linear output for conv transpose
        input_footage = input_footage.view(input_footage.shape[0], 50, 16, 16)
        input_footage = self.conv_dec(input_footage)
        return input_footage


if __name__ == "__main__":
    dummy_data = torch.rand(2, 1, 120, 120)
    model = Analyser()
    start = timer()
    out = model(dummy_data)
    end = timer()
    print(out.shape)
    print(end - start)
