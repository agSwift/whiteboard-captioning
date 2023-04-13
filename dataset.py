"""Contains the StrokeDataset class, which is used to load the stroke data from the .npz files."""
import string
import torch
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
from numpy.lib.npyio import NpzFile

ALL_CHARS = string.ascii_letters + string.digits


class StrokeDataset(Dataset):
    """A PyTorch Dataset class for the stroke data (points and labels)."""

    def __init__(
        self,
        *,
        numbers_data: NpzFile,
        lowercase_data: NpzFile,
        uppercase_data: NpzFile,
    ):
        """Initializes the StrokeDataset class.

        Args:
            numbers_data (NpzFile): The stroke data for the numbers.
            lowercase_data (NpzFile): The stroke data for the lowercase letters.
            uppercase_data (NpzFile): The stroke data for the uppercase letters.

        Raises:
            ValueError: If the stroke data is invalid.
        """
        for data in [numbers_data, lowercase_data, uppercase_data]:
            if "points" not in data or "labels" not in data:
                raise ValueError(
                    f"Invalid {data}. Must contain 'points' and 'labels' keys."
                )

        numbers_points = numbers_data["points"]
        lowercase_points = lowercase_data["points"]
        uppercase_points = uppercase_data["points"]

        # Ensure all stroke points are of equal length, by adding [-1, -1] padding to the end of
        # the shorter stroke points arrays if necessary. This ensures the stroke points arrays can
        # be concatenated into a single array.
        max_num_stroke_points = max(
            numbers_points.shape[1],
            lowercase_points.shape[1],
            uppercase_points.shape[1],
        )

        numbers_points = self._pad_stroke_points(
            numbers_points, max_num_stroke_points
        )
        lowercase_points = self._pad_stroke_points(
            lowercase_points, max_num_stroke_points
        )
        uppercase_points = self._pad_stroke_points(
            uppercase_points, max_num_stroke_points
        )

        self.x = torch.from_numpy(
            np.concatenate(
                (
                    numbers_points,
                    lowercase_points,
                    uppercase_points,
                )
            )
        )

        # Perform one-hot encoding on the labels and concatenate them into a single array.
        numbers_labels = numbers_data["labels"]
        lowercase_labels = lowercase_data["labels"]
        uppercase_labels = uppercase_data["labels"]

        self.y = torch.cat(
            (
                torch.stack(
                    [self._char_to_tensor(char) for char in numbers_labels]
                ),
                torch.stack(
                    [self._char_to_tensor(char) for char in lowercase_labels]
                ),
                torch.stack(
                    [self._char_to_tensor(char) for char in uppercase_labels]
                ),
            )
        )

        assert self.x.shape[0] == self.y.shape[0], (
            f"Number of stroke points ({self.x.shape[0]}) does not match "
            f"number of labels ({self.y.shape[0]})."
        )

    def _pad_stroke_points(
        self, stroke_points: npt.NDArray[np.int_], max_num_stroke_points: int
    ) -> npt.NDArray[np.int_]:
        """Pads the array with [-1, -1] to have a second dimension length of max_num_stroke_points.

        Args:
            stroke_points (npt.NDArray[np.int_]): The array to pad.
            max_num_stroke_points (int): The length to pad the array to.

        Returns:
            The padded array.
        """
        return np.pad(
            stroke_points,
            (
                (0, 0),
                (0, max_num_stroke_points - stroke_points.shape[1]),
                (0, 0),
            ),
            "constant",
            constant_values=-1,
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
