"""Dataset for loading the extracted data from the ISGL dataset."""
import string
import torch
from torch.utils.data import Dataset
from numpy.lib.npyio import NpzFile

ALL_CHARS = string.ascii_letters + string.digits


class StrokeDataset(Dataset):
    """A PyTorch Dataset class for the stroke data (points and labels)."""

    def __init__(
        self,
        stroke_data: NpzFile,
    ):
        """Initializes the StrokeDataset class.

        Args:
            stroke_data (NpzFile): The stroke data to load.

        Raises:
            ValueError: If the stroke data is invalid.
        """
        if not isinstance(stroke_data, NpzFile):
            raise ValueError(
                f"Invalid stroke data: {stroke_data}. "
                f"Must be an instance of {NpzFile}."
            )

        if "points" not in stroke_data or "labels" not in stroke_data:
            raise ValueError(
                f"Invalid stroke data: {stroke_data}. "
                f"Must contain the keys 'points' and 'labels'."
            )

        # Convert the stroke points to a tensor.
        self.x = torch.from_numpy(stroke_data["points"]).float()

        # Perform one-hot encoding on the labels.
        self.y = torch.stack(
            [self._char_to_tensor(char) for char in stroke_data["labels"]]
        )

        assert self.x.shape[0] == self.y.shape[0], (
            f"Number of stroke points ({self.x.shape[0]}) does not match "
            f"number of labels ({self.y.shape[0]})."
        )

    def _char_to_tensor(self, char: str) -> torch.Tensor:
        """Converts a character to a one-hot tensor.

        Args:
            char (str): The character to convert.

        Returns:
            The one-hot tensor.
        """
        tensor = torch.zeros(len(ALL_CHARS))

        char_idx = ALL_CHARS.find(char)
        tensor[char_idx] = 1

        return tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
