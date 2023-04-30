"""Used to extract data from the ISGL dataset.

The 'ISGL Online and Offline Character Recognition' Dataset can be downloaded from:
https://data.mendeley.com/datasets/n7kmd7t7yx/1
"""
import re
from pathlib import Path

from enum import Enum
import numpy as np
import numpy.typing as npt

EXTRACTED_DATA_PATH = Path("data/isgl_data.npz")


class IsglDataDirPath(Enum):
    """An enum class for the ISGL dataset stroke file directory paths."""

    NUMBERS = Path(
        "datasets/ICRGL/ONLINE/CHARACTERS/NUMBER/all image info-number"
    )
    LOWERCASE = Path(
        "datasets/ICRGL/ONLINE/CHARACTERS/LOWER/all image info- lower case"
    )
    UPPERCASE = Path(
        "datasets/ICRGL/ONLINE/CHARACTERS/CAPITAL/capital_all image info"
    )


def _extract_data_from_dir(
    stroke_data_dir_path: IsglDataDirPath,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.str_]]:
    """Extracts the stroke points and labels from the stroke file directory.

    Args:
        stroke_data_dir_path (IsglDataDirPath): The stroke file directory path.

    Returns:
        The stroke points and labels.

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
    max_num_points = 0

    # Go through each stroke file.
    for file_idx, stroke_file in enumerate(stroke_files):
        assert stroke_file.is_file() and stroke_file.suffix == ".txt", (
            f"Invalid stroke file: {stroke_file}. "
            f"Must be a text file with the '.txt' extension."
        )

        num_points = 0
        with open(stroke_file, "r", encoding="utf-8") as file:
            # The first line is the digit/character label.
            file_label = file.readline().rstrip()

            # The remaining lines are potential stroke points.
            file_points = []
            for line in file:
                line = line.rstrip()

                # Check if the line is a point.
                if re.fullmatch(r"[0-9]+\_[0-9]+", line):
                    num_points += 1

                    x_coord, y_coord = line.split("_")
                    file_points.append((int(x_coord), int(y_coord)))

        # Check if the file contains any stroke points.
        if not file_points:
            continue

        stroke_points.append(file_points)
        stroke_labels.append(file_label)

        max_num_points = max(max_num_points, num_points)

    assert len(stroke_points) == len(
        stroke_labels
    ), "The number of stroke points and labels must be equal."

    # Create a numpy array to store the stroke points, where each row is a file.
    # The points are padded with [-1, -1] to make them all the same length.
    stroke_points_arr = np.full(
        (len(stroke_points), max_num_points, 2), -1, dtype=np.float_
    )
    for file_idx, file_points in enumerate(stroke_points):
        stroke_points_arr[file_idx, : len(file_points)] = file_points

    # Convert the stroke labels to a numpy array.
    stroke_labels_arr = np.array(stroke_labels, dtype=np.str_)

    return stroke_points_arr, stroke_labels_arr


def _pad_stroke_points(
    stroke_points: npt.NDArray[np.int_], max_num_stroke_points: int
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


def extract_all_data() -> None:
    """Extracts the stroke data from all stroke file directories, and saves it to an NPZ file.

    Returns:
        None.
    """
    all_points = []
    all_labels = []

    for data_dir_path in IsglDataDirPath:
        points, labels = _extract_data_from_dir(data_dir_path)
        all_points.append(points)
        all_labels.append(labels)

    # Get the maximum number of stroke points across all data directories.
    max_num_stroke_points = max(points.shape[1] for points in all_points)

    # Pad the stroke points arrays.
    all_points = [
        _pad_stroke_points(points, max_num_stroke_points)
        for points in all_points
    ]

    # Concatenate the stroke points and labels arrays.
    all_points = np.concatenate(all_points, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Create a data directory if it doesn't exist.
    Path("data").mkdir(parents=True, exist_ok=True)

    # Save the stroke points and labels to an NPZ file.
    np.savez_compressed(
        str(EXTRACTED_DATA_PATH),
        points=all_points,
        labels=all_labels,
    )

    assert (
        EXTRACTED_DATA_PATH.is_file()
    ), "The stroke points and labels were not saved to an NPZ file."
