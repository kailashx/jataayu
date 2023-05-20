import math

import torch
from torch import nn
from torch.nn import functional as F


class LLTMCell(torch.nn.Module):
    def __init__(self, input_size, state_size):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: int
            Number of channels of input tensor.
        state_size: int
            Number of channels of hidden state.
        """
        super(LLTMCell, self).__init__()
        self.input_features = input_size
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_size + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input_tensor, state):
        """
        Parameters
        ----------
        input_tensor:
            2-D Tensor of shape (b, c)
        state:
            (old_h, old_cell). todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        old_h, old_cell = state
        X = torch.cat([old_h, input_tensor], dim=0)
        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=0)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell

class LLTM(nn.Module):
    def __init__(self, input_size, state_size):
        super(LLTM, self).__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.lltm_layer = LLTMCell(self.input_size, self.state_size)

    def forward(self, input_t, old_state):
        seq_len = input_t.size(1)
        h = []
        for t in range(seq_len):
            new_h, new_c = self.lltm_layer(input_t[t, :], old_state)
            old_state = (new_h, new_c)
            h.append(new_h)
        h = torch.stack(h, dim=1)
        return h

if __name__ == "__main__":
    input = torch.rand(12, 12)
    old_h = torch.rand(12)
    old_cell = torch.rand(12)
    model = LLTM(12, 12)

    output = model(input, (old_h, old_cell))

    print(output.shape)
