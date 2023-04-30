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
import numpy.typing as npt
import scipy.optimize

from tqdm import tqdm

LINE_STROKES_DATA_DIR = Path("datasets/IAM/lineStrokes")
LINE_LABELS_DATA_DIR = Path("datasets/IAM/ascii")

EXTRACTED_DATA_PATH = Path("data/iam_data.npz")


@dataclass
class StrokeData:
    """A dataclass for the stroke data."""

    x_points: list[float]
    y_points: list[float]
    time_stamps: list[float]
    pen_ups: list[int]  # 1 if pen is up (end of stroke), 0 if pen is down.


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


def _parse_stroke_element(stroke_elem: ET.Element) -> StrokeData:
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


def _get_strokes_from_stroke_file(stroke_file: Path) -> list[StrokeData]:
    """Gets the list of stroke data from the stroke file.

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


def _fit_stroke_with_bezier_curve(
    stroke: StrokeData,
) -> npt.NDArray[np.float_]:
    """"Processes the stroke data and fits it with a Bézier curve.
    
    Args:
        stroke (StrokeData): The stroke data.

    Returns:
        A numpy array containing the calculated Bézier curve information.

    Raises:
        ValueError: If the stroke data is invalid.
    """
    # Check if the stroke data is valid.
    if not isinstance(stroke, StrokeData):
        raise ValueError(
            f"Invalid stroke data: {stroke}. Must be an instance of {StrokeData}."
        )

    # If there are less than 2 points, there is no Bézier curve to fit.
    if len(stroke.x_points) < 2:
        return np.zeros((1, 10))

    # Combine x and y points into a single numpy array.
    points = np.column_stack((stroke.x_points, stroke.y_points))

    def bezier_curve(
        s_param: float,
        alpha: npt.NDArray[np.float_],
        beta: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """Calculate the Bézier curve using the given parameters.

        Args:
            s_param (float): The parameter s for the Bézier curve (between 0 and 1).
            alpha (npt.NDArray[np.float_]): The coefficients for the x component of the curve.
            beta (npt.NDArray[np.float_]): The coefficients for the y component of the curve.

        Returns:
            A numpy array containing the x and y coordinates of the Bézier curve at the given parameter s.

        """
        x_bezier_points = (
            alpha[0]
            + alpha[1] * s_param
            + alpha[2] * s_param ** 2
            + alpha[3] * s_param ** 3
        )
        y_bezier_points = (
            beta[0]
            + beta[1] * s_param
            + beta[2] * s_param ** 2
            + beta[3] * s_param ** 3
        )
        return np.array([x_bezier_points, y_bezier_points])

    def objective_function(
        coefficients: npt.NDArray[np.float_], points: npt.NDArray[np.float_]
    ) -> float:
        """Calculate the sum of squared errors (SSE) between Bézier curve points and given points.
        
        Args:
            coefficients (npt.NDArray[np.float_]): The coefficients for the Bézier curve.
            points (npt.NDArray[np.float_]): The points to fit the curve to.
            
        Returns:
            The SSE between the Bézier curve points and the given points."""
        alpha, beta = coefficients[:4], coefficients[4:]
        s_values = np.linspace(0, 1, len(points))
        curve_points = np.array(
            [bezier_curve(s, alpha, beta) for s in s_values]
        )
        errors = curve_points - points
        return np.sum(errors ** 2)

    # Find the coefficients that minimize the SSE.
    initial_coefficients = np.ones(8)
    result = scipy.optimize.minimize(
        objective_function, initial_coefficients, args=(points,)
    )
    alpha, beta = result.x[:4], result.x[4:]

    # Calculate the end point differences and distances.
    end_point_diff = np.array([alpha[3], beta[3]])

    control_point_dist1 = np.linalg.norm(
        np.array([alpha[1], beta[1]])
    ) / np.linalg.norm(end_point_diff)

    control_point_dist2 = np.linalg.norm(
        np.array([alpha[2], beta[2]])
    ) / np.linalg.norm(end_point_diff)

    # Get the angles between control points and endpoints in radians.
    angle1 = math.atan2(beta[1], alpha[1])
    angle2 = math.atan2(beta[2], alpha[2])

    # Calculate the time coefficients.
    time_range = np.max(stroke.time_stamps) - np.min(stroke.time_stamps)
    time_coefficients = np.array(
        [time_range * alpha[1], time_range * alpha[2], time_range * alpha[3]]
    )

    # Determine if the stroke is a pen-up or pen-down curve.
    pen_up_flag = stroke.pen_ups[-1]  # 1 if pen-up, 0 if pen-down.

    return np.array(
        [
            *end_point_diff,
            control_point_dist1,
            control_point_dist2,
            angle1,
            angle2,
            *time_coefficients,
            pen_up_flag,
        ]
    ).reshape(
        1, -1
    )  # Will be a 2D array with shape (1, 10).


def extract_all_data() -> None:
    """Extract all data from the IAM On-Line Handwriting Database and save it to a numpy .npz file.
    
    This function processes the data and saves it in the following format:
    - labels: a list of strings, where each string represents a line label from the database.

    - bezier_data: a 2D numpy array with shape (number_of_strokes, 10), where each row
    represents the Bézier curve information of a stroke. The columns contain the following:
        1. x difference of the end points.
        2. y difference of the end points.
        3. Distance between the first control point and the starting point,
        normalized by the end point differences.
        4. Distance between the second control point and the starting point,
        normalized by the end point differences.
        5. Angle between the first control point and the x-axis in radians.
        6. Angle between the second control point and the x-axis in radians.
        7. Time coefficient for the first control point.
        8. Time coefficient for the second control point.
        9. Time coefficient for the end point.
        10. Pen-up flag (1 if the pen is up after the stroke, 0 if the pen is down).
    """
    all_bezier_curves_data = []
    all_labels = []

    # Get the total number of directories to process.
    total_dirs = sum(1 for _ in os.walk(LINE_STROKES_DATA_DIR))

    stroke_counts = []

    # Go through each directory in the line strokes data directory.
    for root, _, stroke_files in tqdm(
        os.walk(LINE_STROKES_DATA_DIR),
        total=total_dirs,
        desc="Processing directories",
    ):
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
            # Get the label for the stroke file.
            label = _get_label_from_stroke_file(stroke_file)
            if label is None:
                break
            all_labels.append(label)

            # Get the strokes from the stroke file.
            strokes = _get_strokes_from_stroke_file(stroke_file)
            stroke_counts.append(len(strokes))

            # Compute the Bézier curves for the stroke file.
            bezier_curves_data = [
                _fit_stroke_with_bezier_curve(stroke) for stroke in strokes
            ]

            all_bezier_curves_data.append(bezier_curves_data)

    all_labels_arr = np.array(all_labels)

    # Pad the Bezier curves data with -1 so that all strokes have the same number of curves.
    max_num_curves = max(len(curves) for curves in all_bezier_curves_data)
    all_bezier_curves_arr = np.array(
        [
            np.concatenate(
                (curves, np.full((max_num_curves - len(curves), 1, 10), -1),),
                axis=0,
            )
            for curves in all_bezier_curves_data
        ]
    )

    assert (
        all_bezier_curves_arr.shape[0]
        == all_labels_arr.shape[0]
        == len(stroke_counts)
    ), (
        f"Number of stroke counts ({len(stroke_counts)}) "
        f"does not match number of labels ({all_labels_arr.shape[0]}) "
        f"or number of bezier curves ({all_bezier_curves_arr.shape[0]})."
    )

    # Create a data directory if it doesn't exist.
    Path("data").mkdir(parents=True, exist_ok=True)

    # Save the extracted data to an NPZ file.
    np.savez_compressed(
        EXTRACTED_DATA_PATH,
        labels=all_labels_arr,
        bezier_data=all_bezier_curves_arr,
    )
