"""A small, independently testable recurrent cell for LAB 05."""

import torch
from torch import Tensor, nn


class SimpleRNNCell(nn.Module):
    """Compute one tanh recurrent step using explicit weight matrices."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initialize input, recurrent, and bias parameters."""
        super().__init__()
        self.weight_input = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hidden = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_uniform_(self.weight_input)
        nn.init.orthogonal_(self.weight_hidden)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """Return the hidden state after reading one input vector."""
        return torch.tanh(
            x_t @ self.weight_input.T + h_prev @ self.weight_hidden.T + self.bias
        )


def verify_against_torch() -> bool:
    """Compare this cell with ``torch.nn.RNNCell`` using identical parameters."""
    input_size = 2
    hidden_size = 2
    input_vector = torch.tensor([[1.0, 0.0]])
    previous_state = torch.tensor([[0.5, 0.2]])
    input_weights = torch.tensor([[0.5, 0.2], [0.1, 0.4]])
    recurrent_weights = torch.tensor([[0.3, 0.1], [0.2, 0.5]])
    custom_cell = SimpleRNNCell(input_size, hidden_size)
    reference_cell = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
    with torch.no_grad():
        custom_cell.weight_input.copy_(input_weights)
        custom_cell.weight_hidden.copy_(recurrent_weights)
        custom_cell.bias.zero_()
        reference_cell.weight_ih.copy_(input_weights)
        reference_cell.weight_hh.copy_(recurrent_weights)
        reference_cell.bias_ih.zero_()
        reference_cell.bias_hh.zero_()
    return bool(
        torch.allclose(
            custom_cell(input_vector, previous_state),
            reference_cell(input_vector, previous_state),
            atol=1e-7,
        )
    )


if __name__ == "__main__":
    print("Matches torch.nn.RNNCell:", verify_against_torch())
