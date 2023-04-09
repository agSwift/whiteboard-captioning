import re
from pathlib import Path

import numpy as np
import numpy.typing as npt


def extract_preprocess_data(
    stroke_file_dir: Path, stroke_type: str
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Extracts and preprocesses the stroke data from the given stroke file directory.

    Args:
        stroke_file_dir (Path): The directory containing the stroke text files.
        stroke_type (str): The type of stroke data (e.g. "numbers").

    Returns:
        A tuple containing the stroke labels and points.
    """
    # Get all text files in the stroke file directory.
    stroke_files = list(stroke_file_dir.glob("*.txt"))

    stroke_points = []
    stroke_labels = []

    # The maximum number of points seen in a stroke file.
    # This is used to pad the stroke points array.
    max_num_stroke_points = 0

    # Go through each stroke file.
    for file_idx, stroke_file in enumerate(stroke_files):
        assert stroke_file.is_file() and stroke_file.suffix == ".txt"

        num_stroke_points = 0
        with open(stroke_file, "r", encoding="utf-8") as file:
            # The first line is the digit/character label.
            stroke_label = file.readline().rstrip()

            # Go through the rest of the file lines.
            file_points = []
            for line in file:
                line = line.rstrip()

                # Check if the line is a point.
                if re.fullmatch(r"[0-9]+\_[0-9]+", line):
                    num_stroke_points += 1

                    x, y = line.split("_")
                    file_points.append((int(x), int(y)))

        # Check if the file has any stroke points.
        if not file_points:
            continue

        stroke_points.append(file_points)
        stroke_labels.append(stroke_label)

        max_num_stroke_points = max(max_num_stroke_points, num_stroke_points)

    assert len(stroke_points) == len(stroke_labels)

    # Create a numpy array to store the stroke points, where -1 represents a missing point.
    stroke_points_arr = np.full(
        (len(stroke_points), max_num_stroke_points, 2), -1, dtype=np.int_
    )
    for file_idx, file_points in enumerate(stroke_points):
        stroke_points_arr[file_idx, : len(file_points)] = file_points

    # Convert the stroke labels to a numpy array.
    stroke_labels_arr = np.array(stroke_labels, dtype=np.int_)

    # Create the preprocessed data directory if it doesn't exist.
    Path("data").mkdir(parents=True, exist_ok=True)

    # Save the stroke points and labels to an NPZ file.
    np.savez(
        f"data/{stroke_type}",
        points=stroke_points_arr,
        labels=stroke_labels_arr,
    )

    return stroke_labels_arr, stroke_points_arr


if __name__ == "__main__":
    # “ISGL Character Recognition Dataset”: https://data.mendeley.com/datasets/n7kmd7t7yx/1
    numbers_dir = Path(
        "../datasets/ICRGL/ONLINE/CHARACTERS/NUMBER/all image info-number"
    )
    print(extract_preprocess_data(numbers_dir, "numbers"))
