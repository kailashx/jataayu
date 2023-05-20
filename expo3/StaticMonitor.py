import torch
from torch import nn

from timeit import default_timer as timer


class Debug(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        print(x.shape)
        return x


class HAMonitor(nn.Module):
    def __init__(self):
        super(HAMonitor, self).__init__()
        self.layer = nn.Sequential(
            # 640-360
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, padding=(5 // 2, 5 // 2)),
            # Debug(),
            nn.SELU(),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, padding=(5 // 2, 5 // 2)),
            nn.SELU(),
            nn.AdaptiveMaxPool2d((320, 180)),

            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, padding=(5 // 2, 5 // 2)),
            # Debug(),
            nn.SELU(),
            nn.Conv2d(in_channels=15, out_channels=20, kernel_size=5, padding=(5 // 2, 5 // 2)),
            nn.SELU(),
            nn.AdaptiveMaxPool2d((160, 90)),

            nn.Conv2d(in_channels=20, out_channels=25, kernel_size=5, padding=(5 // 2, 5 // 2)),
            # Debug(),
            nn.SELU(),
            nn.Conv2d(in_channels=25, out_channels=20, kernel_size=5, padding=(5 // 2, 5 // 2)),
            nn.SELU(),
            nn.AdaptiveMaxPool2d((88, 72)),

            nn.Conv2d(in_channels=20, out_channels=15, kernel_size=5, padding=(5 // 2, 5 // 2)),
            nn.SELU(),
            nn.Conv2d(in_channels=15, out_channels=10, kernel_size=5, padding=(5 // 2, 5 // 2)),
            nn.SELU(),
            nn.AdaptiveMaxPool2d((44, 36)),

            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=5, padding=(5 // 2, 5 // 2)),
            nn.SELU(),
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=5, padding=(5 // 2, 5 // 2)),
            nn.SELU(),
            nn.AdaptiveMaxPool2d((22, 18)),

            # Debug(),
            nn.Flatten(1),
            # Debug(),

            nn.Linear(22 * 18, 22 * 18),
            nn.SELU(),
            nn.Linear(22 * 18, 16 * 16),
            nn.SELU(),
            nn.Linear(16 * 16, 16 * 16),
            nn.SELU(),
            nn.Linear(16 * 16, 16 * 16),
            nn.SELU(),
            nn.Linear(16 * 16, 16 * 16),
            nn.SELU(),
            nn.Linear(16 * 16, 8*8),
            nn.SELU(),
            nn.Linear(8*8, 8 * 8),
            nn.SELU(),
            nn.Linear(8 * 8, 8 * 8),
            nn.SELU(),
            nn.Linear(8 * 8, 8 * 8),
            nn.SELU(),
            nn.Linear(8 * 8, 4*4),
            nn.SELU(),

        )

    def forward(self, input_t):
        return self.layer(input_t)


if __name__ == "__main__":
    model = StaticMonitor()
    # 480×360
    input_d = torch.rand(1, 640, 360)
    start = timer()
    out = model(input_d)
    end = timer()
    print(out.shape)
    print(out)
    print(f"Time taken by normal model: {end - start}")
