import torch
from torch import nn

from expo1.ConvLSTM import ConvLSTM
from expo1.LLTM import LLTM

'''
x = torch.rand((32, 10, 1, 128, 128))
convlstm = ConvLSTM(1, 16, (3, 3), 1, True, True, False)
_, last_states = convlstm(x)
h = last_states[0][0]  # 0 for layer index, 0 for h index

print(last_states.shape)
print(h)
'''


class JataayuModel(nn.Module):
    def __init__(self):
        super(JataayuModel, self).__init__()
        self.c1 = ConvLSTM(input_dim=1, hidden_dim=1, kernel_size=(5, 5),
                           num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.c2 = ConvLSTM(input_dim=1, hidden_dim=1, kernel_size=(3, 3),
                           num_layers=1, batch_first=True, bias=True, return_all_layers=False)

        self.p1 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 960, 540)),
            nn.LayerNorm([1, 960, 540])
        )

        self.p2 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 480, 270)),
            nn.LayerNorm([1, 480, 270])
        )
        self.p3 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 240, 135)),
            nn.LayerNorm([1, 240, 135])
        )
        self.p4 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 120, 69)),
            nn.LayerNorm([1, 120, 69])
        )
        self.p5 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 60, 36)),
            nn.LayerNorm([1, 60, 36])
        )

        self.p6 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 30, 18)),
            nn.LayerNorm([1, 30, 18]),
            #nn.Flatten(start_dim=2)
        )
        self.fc1 = nn.LSTM(input_size=540, hidden_size=432, num_layers=2, batch_first=True)

        self.fc2 = nn.LSTM(input_size=432, hidden_size=288, num_layers=2, batch_first=True)

        self.fc3 = nn.LSTM(input_size=288, hidden_size=192, num_layers=1, batch_first=True)

        self.fc4 = nn.LSTM(input_size=192, hidden_size=128, num_layers=1, batch_first=True)

        self.fc5 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        self.fc6 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)

        self.fc7 = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True)

        self.out_layer = nn.Sequential(
            nn.Linear(in_features=16, out_features=2),
            nn.Softmax(dim=2)
        )
    def forward(self, x):
        """
        Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
        0 - layer_output_list is the list of lists of length T of each output
        1 - last_state_list is the list of last states
        each element of the list is a tuple (h, c) for hidden state and memory
        """
        # _, last_states = convlstm(x)
        layer_output_list, last_state_list = self.c1(x)
        x = self.p1(layer_output_list[0])

        layer_output_list, last_state_list = self.c1(x)
        x = self.p2(layer_output_list[0])

        layer_output_list, last_state_list = self.c1(x)
        x = self.p3(layer_output_list[0])

        layer_output_list, last_state_list = self.c1(x)
        x = self.p4(layer_output_list[0])

        layer_output_list, last_state_list = self.c2(x)
        x = self.p5(layer_output_list[0])

        layer_output_list, last_state_list = self.c2(x)
        x = self.p6(layer_output_list[0])
        x = torch.flatten(x, start_dim=2)

        x, (_, _) = self.fc1(x)
        x, (_, _) = self.fc2(x)
        x, (_, _) = self.fc3(x)
        x, (_, _) = self.fc4(x)
        x, (_, _) = self.fc5(x)
        x, (_, _) = self.fc6(x)
        x, (_, _) = self.fc7(x)
        x = self.out_layer(x)
        return x


if __name__ == "__main__":
    model = JataayuModel()

    x = torch.rand(32, 10, 1, 128, 128)
    output = model(x)

    print(output.shape)
    print(output[0][0][0:2])
