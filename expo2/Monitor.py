import torch
from torch import nn

from expo1.LLTM import LLTM
from expo1.Quadratic import Quadratic


class VisualCell(nn.Module):
    def __init__(self):
        super(VisualCell, self).__init__()
        # Initial cell state
        self.init_h = self.init_c = torch.nn.init.xavier_normal_(torch.empty(1280 * 720, 1280 * 720), gain=1.0)

        self.l1 = LLTM(1280 * 720, 1280 * 720)
        self.ql2 = Quadratic(1280 * 720, 958*538)
        self.l3 = LLTM(958*538, 958*538)
        self.ql4 = Quadratic(958*538, 480*270)
        self.l5 = LLTM(480*270, 480*270)
        self.ql6 = Quadratic(480*270, 240*135)
        self.l7 = LLTM(240*135, 240*135)
        self.ql8 = Quadratic(240*135, 120*69)
        self.l9 = LLTM(120*69, 120*69)
        self.ql10 = Quadratic(120*69, 60*36)
        self.l11 = LLTM(60*36, 60*36)
        self.ql12 = Quadratic(60*36, 30*18)
        self.l13 = LLTM(30*18, 30*18)
        self.ql14 = Quadratic(30*18, 15*9)
        self.l15 = LLTM(15*9, 15*9)
        self.ql16 = Quadratic(15*9, 8*5)
        self.l17 = LLTM(8*5, 8*5)
        self.ql18 = Quadratic(8*5, 4*3)
        self.l19 = LLTM(4*3, 4*3)
        self.ql20 = Quadratic(2*2, 2)

    def forward(self, input_t):
        seq_len = input_t.size(1)

        for i in range(seq_len):
            input_t, cell = self.l1(input_t, (self.init_h, self.init_c))

        input_t = self.ql2(input_t)
        cell = self.ql2(cell)
        input_t, cell = self.l3(input_t, (cell, ))
        pass

class MedialTemporalLobe(nn.Module):
    def __init__(self):
        super(MedialTemporalLobe, self).__init__()
        # padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.c1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=5,
                padding=(5 // 2, 5 // 2)
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=20,
                out_channels=20,
                kernel_size=5,
                padding=(5 // 2, 5 // 2)
            ),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((960, 540))
        )

    def forward(self, x):
        x = self.c1(x)
        return x


if __name__ == "__main__":
    model = MedialTemporalLobe()
    x = torch.rand(10, 1, 1920, 1080)
    y = model(x)
    print(y.shape)
