import re
from pathlib import Path

import numpy as np
import numpy.typing as npt


def extract_preprocess_data(
    stroke_file_dir: Path,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    # Get all txt files in the stroke file directory.
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

    # Save the stroke labels and points to a numpy file.
    Path("data").mkdir(parents=True, exist_ok=True)

    stroke_labels_data = np.asarray(stroke_labels_arr)
    stroke_points_data = np.asarray(stroke_points_arr)
    np.save("data/stroke_points.npy", stroke_points_data)
    np.save("data/stroke_labels.npy", stroke_labels_data)

    return stroke_labels_arr, stroke_points_arr


if __name__ == "__main__":
    numbers_dir = Path(
        "../datasets/ICRGL/ONLINE/CHARACTERS/NUMBER/all image info-number"
    )
    print(extract_preprocess_data(numbers_dir))
