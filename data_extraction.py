"""Used to extract data from the ISGL dataset.

The 'ISGL Online and Offline Character Recognition' Dataset can be downloaded from:
https://data.mendeley.com/datasets/n7kmd7t7yx/1
"""
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

ISGL_DATA_DIR_PATHS = {
    "numbers": Path(
        "../datasets/ICRGL/ONLINE/CHARACTERS/NUMBER/all image info-number"
    ),
    "lowercase": Path(
        "../datasets/ICRGL/ONLINE/CHARACTERS/LOWER/all image info- lower case"
    ),
    "uppercase": Path(
        "../datasets/ICRGL/ONLINE/CHARACTERS/CAPITAL/capital_all image info"
    ),
}


def extract_isgl_data(
    stroke_type: str,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.str_]]:
    """Extracts the stroke data from the given stroke file directory, and saves it to an NPZ file.

    Args:
        stroke_type (str): The type of stroke data to extract. Must be one of
            'numbers', 'lowercase', or 'uppercase'.

    Returns:
        A tuple containing the stroke points and labels.

    Raises:
        ValueError: If the stroke type is invalid.
    """
    if stroke_type not in ISGL_DATA_DIR_PATHS:
        raise ValueError(
            f"Invalid stroke type: {stroke_type}. "
            f"Must be one of {list(ISGL_DATA_DIR_PATHS.keys())}."
        )

    # Get all text files in the stroke file directory.
    stroke_file_dir = ISGL_DATA_DIR_PATHS[stroke_type]
    stroke_files = list(stroke_file_dir.glob("*.txt"))

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

                    x, y = line.split("_")
                    file_points.append((int(x), int(y)))

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
        (len(stroke_points), max_num_stroke_points, 2), -1, dtype=np.int_
    )
    for file_idx, file_points in enumerate(stroke_points):
        stroke_points_arr[file_idx, : len(file_points)] = file_points

    # Convert the stroke labels to a numpy array.
    stroke_labels_arr = np.array(stroke_labels, dtype=np.str_)

    # Create a data directory if it doesn't exist.
    Path("data").mkdir(parents=True, exist_ok=True)

    # Save the stroke points and labels to an NPZ file.
    np.savez_compressed(
        f"data/{stroke_type}",
        points=stroke_points_arr,
        labels=stroke_labels_arr,
    )

    return stroke_points_arr, stroke_labels_arr
