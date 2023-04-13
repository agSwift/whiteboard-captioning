"""Contains the RNN, GRU, and LSTM models."""
import torch
import torch.nn as nn


class RNN(nn.Module):
    """A recurrent neural network (RNN) model."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
    ):
        """Initializes the RNN model."""

        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x: torch.Tensor):
        """Forward pass of the RNN model."""
        # Set initial hidden states.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )

        # Forward propagate RNN.
        out, _ = self.rnn(x, h0)
        # Decode hidden state of last time step.
        out = self.fc(out[:, -1, :])

        return out


class GRU(nn.Module):
    """A gated recurrent unit (GRU) model."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
    ):
        """Initializes the GRU model."""
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x: torch.Tensor):
        """Forward pass of the GRU model."""
        # Set initial hidden states.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )

        # Forward propagate RNN.
        out, _ = self.gru(x, h0)
        # Decode hidden state of last time step.
        out = self.fc(out[:, -1, :])

        return out


class LSTM(nn.Module):
    """A long short-term memory (LSTM) model."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
    ):
        """Initializes the LSTM model."""
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x: torch.Tensor):
        """Forward pass of the LSTM model."""
        # Set initial hidden states.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )
        # Set initial cell states.
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )

        # Forward propagate RNN.
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step.
        out = self.fc(out[:, -1, :])

        return out
