import torch

from expo1.ConvLSTM import ConvLSTM
"""
x = torch.rand(32, 10, 1, 128, 128)

input_dim=channels,
                 hidden_dim=[64, 64, 128],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True
                 bias=True,
                 return_all_layers=False

convlstm = ConvLSTM(1, 1, (3, 3), 1, True, True, False)
layer_output_list, last_states = convlstm(x)
h = last_states[0][0]  # 0 for layer index, 0 for h index
pool = torch.nn.AdaptiveAvgPool3d((1, 100, 100))
norm = torch.nn.LayerNorm([1, 100, 100])

put = pool(layer_output_list[0])
put = norm(put)
print(put.shape)
"""
