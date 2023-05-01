"""Dataset for loading the extracted data from the IAM dataset."""
import string
import torch
from extraction import DatasetType
from torch.utils.data import Dataset
from numpy.lib.npyio import NpzFile


ALL_CHARS = string.ascii_letters + " "
CHAR_TO_INDEX = {char: index + 1 for index, char in enumerate(ALL_CHARS)}


class StrokeBezierDataset(Dataset):
    """A PyTorch Dataset class for the bezier curve data and labels."""

    def __init__(
        self, all_bezier_data: NpzFile, dataset_type: DatasetType,
    ):
        """Initializes the StrokeBezierDataset class.
        
        Args:
            stroke_bezier_data (NpzFile): All the extracted stroke data.
            dataset_type (DatasetType): The type of dataset to load.
            
        Raises:
            ValueError: If the dataset type is invalid.
            ValueError: If the stroke data is invalid.
        """
        # Check that the dataset type is valid.
        if not isinstance(dataset_type, DatasetType):
            raise ValueError(
                f"Invalid data type: {dataset_type}. "
                f"Must be an instance of {DatasetType}."
            )
        if not isinstance(all_bezier_data, NpzFile):
            raise ValueError(
                f"Invalid stroke data: {all_bezier_data}. "
                f"Must be an instance of {NpzFile}."
            )

        # Check that all_bezier_data contains the required data.
        dataset_type_name = dataset_type.name.lower()
        if f"{dataset_type_name}_bezier_curves" not in all_bezier_data.files:
            raise ValueError(
                f"Invalid stroke data: {all_bezier_data}. "
                f"Must contain {dataset_type_name}_bezier_curves."
            )

        if f"{dataset_type_name}_labels" not in all_bezier_data.files:
            raise ValueError(
                f"Invalid stroke data: {all_bezier_data}. "
                f"Must contain {dataset_type_name}_labels."
            )

        # Load the bezier curves and labels.
        self.x = torch.from_numpy(
            all_bezier_data[f"{dataset_type_name}_bezier_curves"]
        ).float()

        # The maximum length of a label.
        max_label_length = max(
            len(label)
            for label in all_bezier_data[f"{dataset_type_name}_labels"]
        )
        # Encode each label as a tensor of indices.
        self.y = torch.stack(
            [
                self._encode_label(
                    label=label, max_label_length=max_label_length
                )
                for label in all_bezier_data[f"{dataset_type_name}_labels"]
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
