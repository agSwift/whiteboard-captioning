import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    """A recurrent neural network (RNN) model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
    ):
        """Initializes the RNN model."""

        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        """Forward pass of the RNN model."""
        # Set initial hidden states.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            device
        )
        # Forward propagate RNN.
        out, _ = self.rnn(x, h0)

        # Decode hidden state of last time step.
        out = self.fc(out[:, -1, :])

        return out
