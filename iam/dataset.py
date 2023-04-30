"""Dataset for loading the extracted data from the IAM dataset."""
import string
import torch
from torch.utils.data import Dataset
from numpy.lib.npyio import NpzFile

ALL_CHARS = string.ascii_letters + " "
CHAR_TO_INDEX = {char: index + 1 for index, char in enumerate(ALL_CHARS)}


class StrokeBezierDataset(Dataset):
    def __init__(self, stroke_bezier_data: NpzFile):
        """Initializes the StrokeBezierDataset class.
        
        Args:
            stroke_bezier_data (NpzFile): The stroke data to load.
            
        Raises:
            ValueError: If the stroke data is invalid.
        """
        if not isinstance(stroke_bezier_data, NpzFile):
            raise ValueError(
                f"Invalid stroke data: {stroke_bezier_data}. "
                f"Must be an instance of {NpzFile}."
            )

        if (
            "labels" not in stroke_bezier_data
            or "bezier_data" not in stroke_bezier_data
        ):
            raise ValueError(
                f"Invalid stroke data: {stroke_bezier_data}. "
                f"Must contain the keys 'labels' and 'bezier_data'."
            )

        self.x = torch.from_numpy(stroke_bezier_data["bezier_data"]).float()

        # The maximum length of a label.
        max_label_length = max(
            len(label) for label in stroke_bezier_data["labels"]
        )
        # Encode each label as a tensor of indices.
        self.y = torch.stack(
            [
                self._encode_label(
                    label=label, max_label_length=max_label_length
                )
                for label in stroke_bezier_data["labels"]
            ]
        )

        assert self.x.shape[0] == self.y.shape[0], (
            f"Number of bezier data ({self.x.shape[0]}) does not match "
            f"number of labels ({self.y.shape[0]})."
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def _encode_label(
        self, *, label: str, max_label_length: int
    ) -> torch.Tensor:
        """Encodes the given label as a tensor of indices based on CHAR_TO_INDEX.

        Filters out from the label any characters that are not in CHAR_TO_INDEX before
        encoding.
        
        Args:
            label (str): The input label string.

        Returns:
            torch.Tensor: The encoded label as a tensor of indices.
        """
        # Only keep alphabetic characters and spaces in the label.
        label = "".join([char for char in label if char in CHAR_TO_INDEX])

        # Encode each character in the label as an index.
        indices = [CHAR_TO_INDEX[char] for char in label]

        # Pad the label with -1 to make it the same length as the longest label.
        indices += [-1] * (max_label_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)
