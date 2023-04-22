"""Used to extract data from the IAM On-Line Handwriting Database.

The database can be downloaded from:
https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database.

Manually removed files from the dataset:
z01-000z.txt (and it's respective stroke XML files), a08-551z-08.xml, a08-551z-09.xml.
"""

from pathlib import Path
from typing import Optional
import os
import xml.etree.ElementTree as ET

LINE_STROKES_DATA_DIR = Path("../datasets/IAM/lineStrokes")
LINE_LABELS_DATA_DIR = Path("../datasets/IAM/ascii")

EXTRACTED_DATA_PATH = Path("data/iam_data.npz")


def _get_label_from_stroke_file(stroke_file: Path) -> Optional[str]:
    """Gets the line label from the stroke file.

    Args:
        stroke_file (Path): The stroke file path.

    Returns:
        The line label. None if the line label is not found.

    Raises:
        ValueError: If the stroke file is invalid.
        FileNotFoundError: If the stroke file is not found.
        ValueError: If the stroke file is not an XML file.
        ValueError: If the line label index is not a number.
        ValueError: If the line label index is not found.
    """
    # Check if the stroke file is valid.
    if not isinstance(stroke_file, Path):
        raise ValueError(
            f"Invalid stroke file: {stroke_file}. Must be an instance of {Path}."
        )

    if not stroke_file.is_file():
        raise FileNotFoundError(f"Stroke file not found: {stroke_file}.")

    if stroke_file.suffix != ".xml":
        raise ValueError(
            f"Invalid stroke file: {stroke_file}. Must be an XML file."
        )

    # Get the stroke file name without the extension.
    stroke_file_name = stroke_file.stem  # e.g. a01-000u-01

    # Get the line label index.
    line_label_idx = stroke_file_name[-2:]  # e.g. 01
    if not line_label_idx.isdigit():
        raise ValueError(
            f"Invalid stroke file: {stroke_file}. "
            f"Label line index must be a number."
        )
    line_label_idx = int(line_label_idx)

    # Get the labels file directory and name.
    root_labels_dir = stroke_file_name[:3]  # e.g. a01
    sub_labels_dir = stroke_file_name[:7]  # e.g. a01-000
    labels_file_name = stroke_file_name[:-3]  # e.g. a01-000u

    # Get the labels file path. e.g. ../datasets/IAM/ascii/a01/a01-000/a01-000u.txt
    labels_file = (
        LINE_LABELS_DATA_DIR
        / root_labels_dir
        / sub_labels_dir
        / f"{labels_file_name}.txt"
    )
    assert labels_file.is_file(), f"Label file not found: {labels_file}."

    # Get the line label from the label file.
    with open(labels_file, "r", encoding="utf-8") as labels_file:
        # The index of the line where the line labels start (CSR line).
        labels_start_idx = 0

        for i, line in enumerate(labels_file):
            # Check if we have found CSR line and save the index.
            if line.startswith("CSR:"):
                labels_start_idx = i

            if (
                labels_start_idx > 0  # Ensure we have found the CSR line.
                and i
                == labels_start_idx
                + line_label_idx
                + 1  # Add 1 to skip the blank line after the CSR line.
            ):
                # Only keep the characters that are in the IAM dataset.
                return "".join(
                    [char for char in line if char.isalnum() or char == " "]
                )

    # If we have not found the line label, return None.
    return None


def extract_all_data() -> None:
    all_stroke_data = []
    all_labels = []

    # Go through each directory in the line strokes data directory.
    for root, _, stroke_files in os.walk(LINE_STROKES_DATA_DIR):
        # Get all stroke files in the directory.
        stroke_files = [
            # Get the full path to the stroke file.
            Path(root) / Path(file)
            for file in stroke_files
            if file.endswith(".xml")
        ]
        stroke_files.sort()

        # Go through each stroke file in the directory.
        for stroke_file in stroke_files:
            label = _get_label_from_stroke_file(stroke_file)
            if label is None:
                break

            all_labels.append(label)

            # Parse the stroke file.
            stroke_tree = ET.parse(stroke_file)
            stroke_tree_root = stroke_tree.getroot()

            # Go through each stroke in the stroke file.
            for stroke in stroke_tree_root.findall("StrokeSet/Stroke"):
                x_points = []
                y_points = []
                time_stamps = []
                fst_timestamp = 0

                # Go through each point in the stroke.
                for point in stroke.findall("Point"):
                    x_points.append(float(point.attrib["x"]))
                    y_points.append(float(point.attrib["y"]))

                    if not time_stamps:
                        fst_timestamp = float(point.attrib["time"])
                        time_stamps.append(0)
                    else:
                        time_stamps.append(
                            float(point.attrib["time"]) - fst_timestamp
                        )


extract_all_data()
