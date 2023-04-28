"""Used to extract data from the IAM On-Line Handwriting Database.

The database can be downloaded from:
https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database.

The following files were filtered out when extracting the data, as they are invalid or missing data:
a08-551z-08.xml, a08-551z-09.xml, and all z01-000z stroke XML files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os
import xml.etree.ElementTree as ET
import numpy as np

from bezier import Bezier


LINE_STROKES_DATA_DIR = Path("../datasets/IAM/lineStrokes")
LINE_LABELS_DATA_DIR = Path("../datasets/IAM/ascii")

EXTRACTED_DATA_PATH = Path("data/iam_data.npz")


@dataclass
class StrokeData:
    """A dataclass for the stroke data."""

    x_points: list[float]
    y_points: list[float]
    time_stamps: list[float]
    pen_ups: list[int]  # 0 if pen is down, 1 if pen is up (end of stroke).


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


def _parse_stroke_element(stroke: ET.Element,) -> StrokeData:
    """Parses the stroke element.

    Args:
        stroke (ET.Element): The stroke element.

    Returns:
        The stroke data, which contains the x points, y points, time stamps, and pen_ups information.

    Raises:
        ValueError: If the stroke element is invalid.
    """
    # Check if the stroke element is valid.
    if not isinstance(stroke, ET.Element):
        raise ValueError(
            f"Invalid stroke ET element: {stroke}. Must be an instance of {ET.Element}."
        )

    x_points = []
    y_points = []
    time_stamps = []
    pen_ups = []
    fst_timestamp = 0

    # Go through each point in the stroke.
    for i, point in enumerate(stroke.findall("Point")):
        x_points.append(float(point.attrib["x"]))
        y_points.append(float(point.attrib["y"]))

        if not time_stamps:
            # Ensure first time stamp is 0, not the actual time stamp.
            fst_timestamp = float(point.attrib["time"])
            time_stamps.append(0)
        else:
            time_stamps.append(float(point.attrib["time"]) - fst_timestamp)

        if i == len(stroke.findall("Point")) - 1:
            # If this is the last point in the stroke, the pen is up.
            pen_ups.append(1)
        else:
            # If this is not the last point in the stroke, the pen is down.
            pen_ups.append(0)

    assert (
        len(x_points) == len(y_points) == len(time_stamps) == len(pen_ups)
    ), (
        f"Invalid stroke element: {stroke}. "
        "The number of x points, y points, time stamps, and pen ups must be equal."
    )

    return StrokeData(x_points, y_points, time_stamps, pen_ups)


def _get_data_from_stroke_file(stroke_file: Path,) -> list[StrokeData]:
    """Gets the data from the stroke file.

    Args:
        stroke_file (Path): The stroke file path.

    Returns:
        The list of stroke data.

    Raises:
        ValueError: If the stroke file is invalid.
        FileNotFoundError: If the stroke file is not found.
        ValueError: If the stroke file is not an XML file.
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

    tree = ET.parse(stroke_file)
    root = tree.getroot()
    stroke_set = root.find("StrokeSet")

    strokes = [
        _parse_stroke_element(stroke)
        for stroke in stroke_set.findall("Stroke")
    ]

    # Normalize the strokes.
    max_y = float(
        root.find("WhiteboardDescription/DiagonallyOppositeCoords").attrib["y"]
    )

    for stroke in strokes:
        min_x = min(stroke.x_points)
        min_y = min(stroke.y_points)

        # Shift x and y points to start at 0.
        stroke.x_points = [x - min_x for x in stroke.x_points]
        stroke.y_points = [y - min_y for y in stroke.y_points]

        # Scale points so that the y_points are between 0 and 1. The x_points will be scaled
        # by the same amount, to preserve the aspect ratio.
        scale = (
            1 if max_y - min_y == 0 else 1 / (max_y - min_y)
        )  # Avoid division by 0.

        stroke.x_points = [x * scale for x in stroke.x_points]
        stroke.y_points = [y * scale for y in stroke.y_points]

        assert all(0 <= y <= 1 for y in stroke.y_points), (
            f"Invalid stroke file: {stroke_file}. "
            f"All y-points must be between 0 and 1."
        )

    assert all(
        len(stroke.x_points)
        == len(stroke.y_points)
        == len(stroke.time_stamps)
        == len(stroke.pen_ups)
        for stroke in strokes
    ), (
        f"Invalid stroke file: {stroke_file}. "
        "The number of x points, y points, time stamps, and pen ups must be equal."
    )

    return strokes


def _process_bezier_curve(
    stroke: StrokeData, num_points: int = 100
) -> StrokeData:
    """Process the Bezier curve for the given stroke.

    Args:
        stroke (StrokeData): The stroke data.
        num_points (int): The number of points to sample along the Bezier curve.

    Returns:
        The processed stroke data containing the Bezier curve points.
    """
    if len(stroke.x_points) < 2:
        return stroke

    points = np.column_stack((stroke.x_points, stroke.y_points))
    bezier = Bezier(points)
    t_values = np.linspace(0, 1, num_points)
    bezier_points = bezier.get_points(t_values)

    x_points = list(bezier_points[:, 0])
    y_points = list(bezier_points[:, 1])
    time_stamps = np.interp(
        t_values,
        np.linspace(0, 1, len(stroke.time_stamps)),
        stroke.time_stamps,
    ).tolist()

    return StrokeData(x_points, y_points, time_stamps, stroke.end_stroke)


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
            # Don't include these files. They are invalid or missing data.
            and not file.startswith("z01-000z")
            and not file.startswith("a08-551z-08")
            and not file.startswith("a08-551z-09")
        ]
        stroke_files.sort()

        # Go through each stroke file in the directory.
        for stroke_file in stroke_files:
            # Get the label from the stroke file.
            label = _get_label_from_stroke_file(stroke_file)
            if label is None:
                break
            all_labels.append(label)

            # Get the data from the stroke file.
            strokes = _get_data_from_stroke_file(stroke_file)

            bezier_curves = [
                _process_bezier_curve(stroke) for stroke in strokes
            ]


extract_all_data()
