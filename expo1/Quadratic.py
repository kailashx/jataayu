import math

import torch
from torch import nn
from torch.nn import functional as F


class Quadratic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Quadratic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weights = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(self.output_size, self.input_size), gain=1.0))
        self.bias = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(1, self.output_size), gain=1.0))

    def forward(self, data):
        # print(self.weights)
        # self.weights = torch.transpose(self.weights, 0, 1)

        data = torch.pow(F.elu(data), 2)
        y = torch.matmul(self.weights, data)
        y = torch.add(y, self.bias)
        return y


if __name__ == "__main__":
    model = Quadratic(22 * 18, 16 * 9)
    features = torch.rand(1, 396)
    out = model(features.flatten(start_dim=0))
    print(features)
    print(out)
    print(out.shape)
