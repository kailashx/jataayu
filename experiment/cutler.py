import torch
from torch import nn

model = torch.load(r'E:\research\superai\cutler_cascade_final.pth', map_location=torch.device('cpu'))

class CutLer(nn.Module):
    def __init__(self):

input = torch.randn(1900, 1080, 5)

output = model(input)

print(output)

