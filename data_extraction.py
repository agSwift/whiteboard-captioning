"""Used to extract data from the ISGL dataset.

The 'ISGL Online and Offline Character Recognition' Dataset can be downloaded from:
https://data.mendeley.com/datasets/n7kmd7t7yx/1
"""
import re
from pathlib import Path

from enum import Enum
import numpy as np
import numpy.typing as npt


class IsglDataDirPath(Enum):
    """An enum class for the ISGL dataset stroke file directory paths."""

    NUMBERS = Path(
        "../datasets/ICRGL/ONLINE/CHARACTERS/NUMBER/all image info-number"
    )
    LOWERCASE = Path(
        "../datasets/ICRGL/ONLINE/CHARACTERS/LOWER/all image info- lower case"
    )
    UPPERCASE = Path(
        "../datasets/ICRGL/ONLINE/CHARACTERS/CAPITAL/capital_all image info"
    )


def _extract_data_from_dir_to_npz(
    stroke_data_dir_path: IsglDataDirPath,
) -> None:
    """Extracts the stroke data from the given stroke file directory, and saves it to an NPZ file.

    Args:
        stroke_data_dir_path (IsglDataDirPath): The stroke file directory path.

    Returns:
        None.

    Raises:
        ValueError: If the stroke data directory path is invalid.
    """
    if not isinstance(stroke_data_dir_path, IsglDataDirPath):
        raise ValueError(
            f"Invalid stroke data directory path: {stroke_data_dir_path}. "
            f"Must be an instance of {IsglDataDirPath}."
        )

    # Get all text files in the stroke file directory.
    stroke_files = list(stroke_data_dir_path.value.glob("*.txt"))

    stroke_points = []
    stroke_labels = []

    # The maximum number of points seen in a stroke file.
    # This is used to pad the stroke points array.
    max_num_stroke_points = 0

    # Go through each stroke file.
    for file_idx, stroke_file in enumerate(stroke_files):
        assert stroke_file.is_file() and stroke_file.suffix == ".txt", (
            f"Invalid stroke file: {stroke_file}. "
            f"Must be a text file with the '.txt' extension."
        )

        num_stroke_points = 0
        with open(stroke_file, "r", encoding="utf-8") as file:
            # The first line is the digit/character label.
            stroke_label = file.readline().rstrip()

            # The remaining lines are potential stroke points.
            file_points = []
            for line in file:
                line = line.rstrip()

                # Check if the line is a point.
                if re.fullmatch(r"[0-9]+\_[0-9]+", line):
                    num_stroke_points += 1

                    x_coord, y_coord = line.split("_")
                    file_points.append((int(x_coord), int(y_coord)))

        # Check if the file contains any stroke points.
        if not file_points:
            continue

        stroke_points.append(file_points)
        stroke_labels.append(stroke_label)

        max_num_stroke_points = max(max_num_stroke_points, num_stroke_points)

    assert len(stroke_points) == len(
        stroke_labels
    ), "The number of stroke points and labels must be equal."

    # Create a numpy array to store the stroke points, where each row is a file.
    # The points are padded with [-1, -1] to make them all the same length.
    stroke_points_arr = np.full(
        (len(stroke_points), max_num_stroke_points, 2), -1, dtype=np.float32
    )
    for file_idx, file_points in enumerate(stroke_points):
        stroke_points_arr[file_idx, : len(file_points)] = file_points

    # Convert the stroke labels to a numpy array.
    stroke_labels_arr = np.array(stroke_labels, dtype=np.str_)

    # Create a data directory if it doesn't exist.
    Path("data").mkdir(parents=True, exist_ok=True)

    # Save the stroke points and labels to an NPZ file.
    np.savez_compressed(
        f"data/{stroke_data_dir_path.name.lower()}",
        points=stroke_points_arr,
        labels=stroke_labels_arr,
    )


def extract_all_data() -> None:
    for data_dir_path in IsglDataDirPath:
        _extract_data_from_dir_to_npz(data_dir_path)
