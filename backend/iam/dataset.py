"""Dataset for loading the extracted data from the IAM dataset."""
import torch
from torch.utils.data import Dataset
from numpy.lib.npyio import NpzFile

from iam import extraction


class StrokeBezierDataset(Dataset):
    """A PyTorch Dataset class for the bezier curve data and labels."""

    def __init__(
        self,
        all_bezier_data: NpzFile,
        dataset_type: extraction.DatasetType,
    ):
        """Initializes the StrokeBezierDataset class.

        Args:
            stroke_bezier_data (NpzFile): All the extracted stroke data.
            dataset_type (extraction.DatasetType): The type of dataset to load.

        Raises:
            ValueError: If the dataset type is invalid.
            ValueError: If the stroke data is invalid.
        """
        # Check that the dataset type is valid.
        if not isinstance(dataset_type, extraction.DatasetType):
            raise ValueError(
                f"Invalid data type: {dataset_type}. "
                f"Must be an instance of {extraction.DatasetType}."
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
        if "all_chars" not in all_bezier_data.files:
            raise ValueError(
                f"Invalid stroke data: {all_bezier_data}. "
                f"Must contain all_chars."
            )
        all_chars = all_bezier_data["all_chars"]
        self._char_to_index = {
            char: index + 1 for index, char in enumerate(all_chars)
        }

        # Load the bezier curves and labels.
        self.all_bezier_curves = torch.from_numpy(
            all_bezier_data[f"{dataset_type_name}_bezier_curves"]
        ).float()
        self.max_num_bezier_curves = self.all_bezier_curves.shape[1]

        # The length of each label, prior to padding.
        self.target_lengths = torch.tensor(
            [
                len(label)
                for label in all_bezier_data[f"{dataset_type_name}_labels"]
            ],
            dtype=torch.long,
        )

        # The maximum length of a label.
        max_label_length = max(
            len(label)
            for label in all_bezier_data[f"{dataset_type_name}_labels"]
        )
        # Encode each label as a tensor of indices.
        self.labels = torch.stack(
            [
                self._encode_label(
                    label=label, max_label_length=max_label_length
                )
                for label in all_bezier_data[f"{dataset_type_name}_labels"]
            ]
        )

        assert self.all_bezier_curves.shape[0] == self.labels.shape[0], (
            f"Number of bezier data ({self.all_bezier_curves.shape[0]}) does not match "
            f"number of labels ({self.labels.shape[0]})."
        )

    def __len__(self):
        return len(self.all_bezier_curves)

    def __getitem__(self, idx):
        return self.all_bezier_curves[idx], self.labels[idx]

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
        # Filter out any characters that are not in CHAR_TO_INDEX.
        label = "".join(
            [char for char in label if char in self._char_to_index]
        )

        # Encode each character in the label as an index.
        indices = [self._char_to_index[char] for char in label]

        # Pad the label with -1 to make it the same length as the longest label.
        indices += [-1] * (max_label_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def get_index_to_char_mapping(self):
        """Returns the index to character mapping.

        Returns:
            dict: The index to character mapping.
        """
        index_to_char = {
            index: char for char, index in self._char_to_index.items()
        }
        index_to_char[0] = "_"  # Epislon character.

        return index_to_char
