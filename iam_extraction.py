"""Used to extract data from the IAM On-Line Handwriting Database.

The database can be downloaded from:
https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database.

The following files were filtered out when extracting the data, as they are invalid or missing data:
a08-551z-08.xml, a08-551z-09.xml, and all z01-000z stroke XML files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize


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


def _get_line_from_labels_file(
    labels_file: Path, line_idx: int
) -> Optional[str]:
    """Gets the line label from the labels file.

    Args:
        labels_file (Path): The labels file path.
        line_idx (int): The line index.

    Returns:
        The line label. None if the line label is not found.

    Raises:
        ValueError: If the labels file is invalid.
        FileNotFoundError: If the labels file is not found.
        ValueError: If the labels file is not a file.
        ValueError: If the line label index is not a number.
    """
    # Check if the labels file is valid.
    if not isinstance(labels_file, Path):
        raise ValueError(
            f"Invalid labels file: {labels_file}. Must be an instance of {Path}."
        )

    # Check if the labels file exists.
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}.")

    # Check if the labels file is a file.
    if not labels_file.is_file():
        raise ValueError(f"Labels file is not a file: {labels_file}.")

    # Check if the line label index is valid.
    if not isinstance(line_idx, int):
        raise ValueError(
            f"Invalid line label index: {line_idx}. Must be an integer."
        )

    # Get the line label from the labels file.
    with open(labels_file, "r", encoding="utf-8") as file:
        # The index of the line where the line labels start (CSR line).
        labels_start_idx = 0

        for i, line in enumerate(file):
            # Check if we have found CSR line and save the index.
            if line.startswith("CSR:"):
                labels_start_idx = i

            if (
                labels_start_idx > 0  # Ensure we have found the CSR line.
                and i
                == labels_start_idx
                + line_idx
                + 1  # Add 1 to skip the blank line after the CSR line.
            ):
                # Only keep the characters that are in the IAM dataset.
                return "".join(
                    [char for char in line if char.isalnum() or char == " "]
                )

    # If we have not found the line label, return None.
    return None


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

    # Get the line label from the labels file.
    return _get_line_from_labels_file(labels_file, line_label_idx)


def _parse_stroke_element(stroke_elem: ET.Element,) -> StrokeData:
    """Parses the stroke element.

    Args:
        stroke (ET.Element): The stroke element.

    Returns:
        The stroke data, which contains the x points, y points, time stamps, and pen_ups information.

    Raises:
        ValueError: If the stroke element is invalid.
    """
    # Check if the stroke element is valid.
    if not isinstance(stroke_elem, ET.Element):
        raise ValueError(
            f"Invalid stroke ET element: {stroke_elem}. Must be an instance of {ET.Element}."
        )

    x_points = []
    y_points = []
    time_stamps = []
    pen_ups = []
    fst_timestamp = 0

    # Go through each point in the stroke.
    for i, point in enumerate(stroke_elem.findall("Point")):
        x_points.append(float(point.attrib["x"]))
        y_points.append(float(point.attrib["y"]))

        if not time_stamps:
            # Ensure first time stamp is 0, not the actual time stamp.
            fst_timestamp = float(point.attrib["time"])
            time_stamps.append(0)
        else:
            time_stamps.append(float(point.attrib["time"]) - fst_timestamp)

        if i == len(stroke_elem.findall("Point")) - 1:
            # If this is the last point in the stroke, the pen is up.
            pen_ups.append(1)
        else:
            # If this is not the last point in the stroke, the pen is down.
            pen_ups.append(0)

    assert (
        len(x_points) == len(y_points) == len(time_stamps) == len(pen_ups)
    ), (
        f"Invalid stroke element: {stroke_elem}. "
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


# TODO: Clean up this function and code in general. Add type hints, docstrings, etc.
def _process_bezier_curve(stroke: StrokeData) -> np.ndarray:
    if len(stroke.x_points) < 2:
        return stroke

    points = np.column_stack((stroke.x_points, stroke.y_points))

    # Define the Bézier curve as a function of the parameter s and coefficients (alpha, beta)
    def bezier_curve(
        s: float, alpha: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        x = alpha[0] + alpha[1] * s + alpha[2] * s ** 2 + alpha[3] * s ** 3
        y = beta[0] + beta[1] * s + beta[2] * s ** 2 + beta[3] * s ** 3
        return np.array([x, y])

    # Define the objective function to minimize the sum of squared errors (SSE)
    def objective_function(
        coefficients: np.ndarray, points: np.ndarray
    ) -> float:
        alpha, beta = coefficients[:4], coefficients[4:]
        s_values = np.linspace(0, 1, len(points))
        curve_points = np.array(
            [bezier_curve(s, alpha, beta) for s in s_values]
        )
        errors = curve_points - points
        return np.sum(errors ** 2)

    # Find the coefficients that minimize the SSE
    initial_coefficients = np.ones(8)
    result = scipy.optimize.minimize(
        objective_function, initial_coefficients, args=(points,)
    )
    alpha, beta = result.x[:4], result.x[4:]

    # Get the vectors between the end points
    dx_dy = np.array([alpha[3], beta[3]])

    # Get the distances between the end points and the control points
    d1 = np.linalg.norm(np.array([alpha[1], beta[1]]))
    d2 = np.linalg.norm(np.array([alpha[2], beta[2]]))
    d1 /= np.linalg.norm(dx_dy)
    d2 /= np.linalg.norm(dx_dy)

    # Get the angles between control points and endpoints in radians
    a1 = math.atan2(beta[1], alpha[1])
    a2 = math.atan2(beta[2], alpha[2])

    # Calculate the time coefficients
    time_range = np.max(stroke.time_stamps) - np.min(stroke.time_stamps)
    gamma = np.array(
        [time_range * alpha[1], time_range * alpha[2], time_range * alpha[3]]
    )

    # Boolean value indicating whether this is a pen-up or pen-down curve
    p = int(stroke.pen_ups[-1])

    result = np.array([*dx_dy, d1, d2, a1, a2, *gamma, p])
    print(result.shape)
    print(result)

    # Plot the result
    s_values = np.linspace(0, 1, len(points))
    curve_points = np.array([bezier_curve(s, alpha, beta) for s in s_values])
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.plot(curve_points[:, 0], curve_points[:, 1], "-")
    plt.show()

    return np.array([*dx_dy, d1, d2, a1, a2, *gamma, p])


# TODO: Finish implementing this function.
def extract_all_data() -> None:
    all_labels = []
    i = 0

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

            _process_bezier_curve(strokes[0])
            # plot the strokes[0]
            plt.plot(strokes[0].x_points, strokes[0].y_points)
            plt.show()

            print(label)

            # bezier_curves = [
            #     _process_bezier_curve(stroke) for stroke in strokes
            # ]


extract_all_data()
