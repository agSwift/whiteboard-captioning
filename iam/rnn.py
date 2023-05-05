"""Contains the RNN, GRU, and LSTM models."""
from enum import Enum

import torch
import torch.nn as nn


class RNNType(Enum):
    """Enum for the RNN model type."""

    RNN = "RNN"
    LSTM = "LSTM"
    GRU = "GRU"


class BaseModel(nn.Module):
    """Base class for RNN, LSTM, and GRU models."""

    def __init__(
        self,
        *,
        bezier_curve_dimension: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int,
        rnn_type: RNNType,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Output linear layer for logits.
        self.fc = nn.Linear(
            hidden_size, num_classes + 1
        )  # Account for the blank symbol in CTC.

        if rnn_type == RNNType.RNN:
            self.rnn = nn.RNN(bezier_curve_dimension, hidden_size, num_layers,)
        elif rnn_type == RNNType.LSTM:
            self.rnn = nn.LSTM(
                bezier_curve_dimension, hidden_size, num_layers,
            )
        elif rnn_type == RNNType.GRU:
            self.rnn = nn.GRU(bezier_curve_dimension, hidden_size, num_layers,)
        else:
            raise ValueError(
                f"Invalid RNN type: {rnn_type}. Must be one of {RNNType}."
            )

        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): The input tensor of shape 
                (num_bezier_curves, batch_size, bezier_curve_dimension).
        
        Returns:
            torch.Tensor: The output tensor of shape
                (batch_size, num_classes + 1, num_bezier_curves).
        """
        out, _ = self.rnn(x)

        out = self.fc(
            out
        )  # Apply the fully connected layer to all time steps.
        out = torch.nn.functional.log_softmax(
            out, dim=2
        )  # Apply log_softmax to the output.
        return out


class RNN(BaseModel):
    """Recurrent Neural Network (RNN) model."""

    def __init__(
        self,
        *,
        bezier_curve_dimension: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            bezier_curve_dimension=bezier_curve_dimension,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            rnn_type=RNNType.RNN,
            device=device,
        )


class LSTM(BaseModel):
    """Long Short-Term Memory (LSTM) model."""

    def __init__(
        self,
        *,
        bezier_curve_dimension: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            bezier_curve_dimension=bezier_curve_dimension,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            rnn_type=RNNType.LSTM,
            device=device,
        )


class GRU(BaseModel):
    """Gated Recurrent Unit (GRU) model."""

    def __init__(
        self,
        *,
        bezier_curve_dimension: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            bezier_curve_dimension=bezier_curve_dimension,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            rnn_type=RNNType.GRU,
            device=device,
        )
