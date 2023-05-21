"""Used to extract data from the IAM On-Line Handwriting Database.

The database can be downloaded from:
https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database.

The following files were filtered out when extracting the data, as they are invalid or missing data:
a08-551z-08.xml, a08-551z-09.xml, and all z01-000z stroke XML files.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import math
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# import scipy.optimize
from scipy.special import comb
from tqdm import tqdm

LINE_STROKES_DATA_DIR = Path("datasets/IAM/lineStrokes")
LINE_LABELS_DATA_DIR = Path("datasets/IAM/ascii")

EXTRACTED_DATA_PATH = Path("data/iam_bernstein_data.npz")


class DatasetType(Enum):
    """An enum for the dataset types."""

    TRAIN = Path("datasets/IAM/trainset.txt")
    VAL_1 = Path("datasets/IAM/valset1.txt")
    VAL_2 = Path("datasets/IAM/valset2.txt")
    TEST = Path("datasets/IAM/testset.txt")


@dataclass
class BezierData:
    """A dataclass for the bezier curve data."""

    label_file_names: set[str]
    bezier_curves_data: list[list[npt.NDArray[np.float_]]] = field(
        default_factory=list
    )
    labels: list[str] = field(default_factory=list)


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


def _get_label_from_stroke_file(
    stroke_file: Path,
) -> tuple[Optional[str], str]:
    """Gets the line label and the labels file name that the label belongs to.

    Args:
        stroke_file (Path): The stroke file path.

    Returns:
        The line label and the label file name. None if the line label is not found.

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
    return (
        _get_line_from_labels_file(labels_file, line_label_idx),
        labels_file_name,
    )


def _parse_stroke_element(stroke_elem: ET.Element) -> StrokeData:
    """Parses the stroke element.

    Args:
        stroke (ET.Element): The stroke element.

    Returns:
        The stroke data, which contains the x points, y points, time stamps,
        and pen_ups information.

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


def get_bezier_parameters(
    x_points: list[float], y_points: list[float], degree: int = 3
) -> list[list[float]]:
    """Compute least square bezier curve fit using penrose pseudoinverse.

    Args:
        x_points (list[float]): X-coordinates of the points to fit.
        y_points (list[float]): Y-coordinates of the points to fit.
        degree (int, optional): Degree of the Bézier curve. Defaults to 3.

    Returns:
        List[List[float]]: Parameters of the Bézier curve.

    Raises:
        ValueError: If the degree is less than 1.
        ValueError: If the number of x points and y points are not equal.
        ValueError: If the number of points is less than the degree + 1.
    """
    if degree < 1:
        raise ValueError(
            f"Invalid degree: {degree}. Degree must be at least 1."
        )

    if len(x_points) != len(y_points):
        raise ValueError(
            f"Invalid points: {x_points}, {y_points}. "
            "The number of x points and y points must be equal."
        )

    if len(x_points) < degree + 1:
        raise ValueError(
            f"Invalid points: {x_points}, {y_points}. "
            f"The number of points must be at least the {degree + 1} (degree + 1)."
        )

    def bernstein_poly(n: int, t: float, k: int) -> float:
        """Bernstein polynomial when a = 0 and b = 1."""
        return t**k * (1 - t) ** (n - k) * comb(n, k)

    def bernstein_matrix(T: list[float]) -> np.matrix:
        """Bernstein matrix for Bézier curves."""
        return np.matrix(
            [
                [bernstein_poly(degree, t, k) for k in range(degree + 1)]
                for t in T
            ]
        )

    def least_square_fit(points: np.ndarray, M: np.matrix) -> np.matrix:
        """Perform least square fit using pseudoinverse."""
        M_pseudoinv = np.linalg.pinv(M)
        return M_pseudoinv * points

    T = np.linspace(0, 1, len(x_points))
    M = bernstein_matrix(T)
    points = np.array(list(zip(x_points, y_points)))

    bezier_params = least_square_fit(points, M).tolist()
    bezier_params[0] = [x_points[0], y_points[0]]
    bezier_params[-1] = [x_points[-1], y_points[-1]]
    return bezier_params


def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=50):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [ [1,1],
              [2,3],
              [4,5], ..[Xn, Yn] ]
     nTimes is the number of time steps, defaults to 1000

     See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)]
    )

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def _fit_stroke_with_bezier_curve(
    stroke: StrokeData,
) -> npt.NDArray[np.float_]:
    """Processes the stroke data and fits it with a Bezier curve.

    Args:
        stroke (StrokeData): The stroke data.

    Returns:
        A numpy array containing the calculated Bezier curve information.

    Raises:
        ValueError: If the stroke data is invalid.
    """
    # Check if the stroke data is valid.
    if not isinstance(stroke, StrokeData):
        raise ValueError(
            f"Invalid stroke data: {stroke}. Must be an instance of {StrokeData}."
        )

    # If there are less than 2 points, there is no Bezier curve to fit.
    if len(stroke.x_points) < 4:
        return np.zeros((1, 10))

    # Get the Bezier parameters for the stroke data.
    bezier_params = get_bezier_parameters(
        stroke.x_points, stroke.y_points, degree=3
    )

    # Compute the fitted points from the Bezier parameters.
    fitted_points = bezier_curve(bezier_params)

    # Compute the distance between end points
    end_point_diff = np.array(
        [
            bezier_params[-1][0] - bezier_params[0][0],
            bezier_params[-1][1] - bezier_params[0][1],
        ]
    )
    end_point_diff_norm = max(np.linalg.norm(end_point_diff), 0.001)

    # Compute the distance between the first control point and the starting point
    control_point_dist1 = (
        np.linalg.norm(
            np.array(
                [
                    bezier_params[1][0] - bezier_params[0][0],
                    bezier_params[1][1] - bezier_params[0][1],
                ]
            )
        )
        / end_point_diff_norm
    )

    # Compute the distance between the second control point and the starting point
    control_point_dist2 = (
        np.linalg.norm(
            np.array(
                [
                    bezier_params[2][0] - bezier_params[0][0],
                    bezier_params[2][1] - bezier_params[0][1],
                ]
            )
        )
        / end_point_diff_norm
    )

    # Get the angles between control points and end points in radians.
    angle1 = math.atan2(
        bezier_params[1][1] - bezier_params[0][1],
        bezier_params[1][0] - bezier_params[0][0],
    )
    angle2 = math.atan2(
        bezier_params[2][1] - bezier_params[0][1],
        bezier_params[2][0] - bezier_params[0][0],
    )

    # Calculate the time coefficients
    time_range = np.max(stroke.time_stamps) - np.min(stroke.time_stamps)
    time_coefficients = np.array(
        [
            time_range * bezier_params[1][0],
            time_range * bezier_params[2][0],
            time_range * bezier_params[3][0],
        ]
    )

    # Determine if the stroke is a pen-up or pen-down curve.
    pen_up_flag = stroke.pen_ups[-1]  # 1 if pen-up, 0 if pen-down.

    # Plot the original points
    # plt.plot(stroke.x_points, stroke.y_points, "ro",label='Original Points')
    # # Get the Bezier parameters based on a degree.
    # data = get_bezier_parameters(stroke.x_points, stroke.y_points, degree=3)
    # x_val = [x[0] for x in data]
    # y_val = [x[1] for x in data]
    # print(data)
    # # Plot the control points
    # plt.plot(x_val,y_val,'k--o', label='Control Points')
    # # Plot the resulting Bezier curve
    # xvals, yvals = bezier_curve(data, nTimes=1000)
    # plt.plot(xvals, yvals, 'b-', label='B Curve')
    # plt.legend()
    # plt.show()

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


# def _fit_stroke_with_bezier_curve(
#     stroke: StrokeData,
# ) -> npt.NDArray[np.float_]:
#     """Processes the stroke data and fits it with a Bezier curve.

#     Args:
#         stroke (StrokeData): The stroke data.

#     Returns:
#         A numpy array containing the calculated Bezier curve information.

#     Raises:
#         ValueError: If the stroke data is invalid.
#     """
#     # Check if the stroke data is valid.
#     if not isinstance(stroke, StrokeData):
#         raise ValueError(
#             f"Invalid stroke data: {stroke}. Must be an instance of {StrokeData}."
#         )

#     # If there are less than 2 points, there is no Bezier curve to fit.
#     if len(stroke.x_points) < 2:
#         return np.zeros((1, 10))

#     # Combine x and y points into a single numpy array.
#     stroke_points = np.column_stack((stroke.x_points, stroke.y_points))

#     def bezier_curve(
#         s_param: float,
#         alpha: npt.NDArray[np.float_],
#         beta: npt.NDArray[np.float_],
#     ) -> npt.NDArray[np.float_]:
#         """Calculate the Bezier curve using the given parameters.

#         Args:
#             s_param (float): The parameter s for the Bezier curve (between 0 and 1).
#             alpha (npt.NDArray[np.float_]): The coefficients for the x component of the curve.
#             beta (npt.NDArray[np.float_]): The coefficients for the y component of the curve.

#         Returns:
#             A numpy array containing the x and y coordinates of the Bezier curve at
#             the given parameter s.
#         """
#         x_bezier_point = (
#             alpha[0]
#             + alpha[1] * s_param
#             + alpha[2] * s_param**2
#             + alpha[3] * s_param**3
#         )
#         y_bezier_point = (
#             beta[0]
#             + beta[1] * s_param
#             + beta[2] * s_param**2
#             + beta[3] * s_param**3
#         )
#         return np.array([x_bezier_point, y_bezier_point])

#     def objective_function(
#         coefficients: npt.NDArray[np.float_], points: npt.NDArray[np.float_]
#     ) -> float:
#         """Calculate the sum of squared errors (SSE) between Bezier curve points and given points.

#         Args:
#             coefficients (npt.NDArray[np.float_]): The coefficients for the Bezier curve.
#             points (npt.NDArray[np.float_]): The points to fit the curve to.

#         Returns:
#             The SSE between the Bezier curve points and the given points.
#         """
#         alpha, beta = coefficients[:4], coefficients[4:]
#         s_values = np.linspace(0, 1, len(points))
#         curve_points = np.array(
#             [bezier_curve(s, alpha, beta) for s in s_values]
#         )
#         errors = curve_points - points
#         return np.sum(errors**2)

#     # Find the coefficients that minimize the SSE.
#     initial_coefficients = np.ones(8)
#     result = scipy.optimize.minimize(
#         objective_function, initial_coefficients, args=(stroke_points,)
#     )
#     alpha, beta = result.x[:4], result.x[4:]

#     # Distance between end points.
#     end_point_diff = np.array([alpha[3], beta[3]])
#     end_point_diff_norm = max(np.linalg.norm(end_point_diff), 0.001)

#     # Distance between first control point and starting point,
#     # normalized by the distance between end points.
#     control_point_dist1 = (
#         np.linalg.norm(np.array([alpha[1], beta[1]])) / end_point_diff_norm
#     )

#     # Distance between second control point and starting point,
#     # normalized by the distance between end points.
#     control_point_dist2 = (
#         np.linalg.norm(np.array([alpha[2], beta[2]])) / end_point_diff_norm
#     )

#     # Get the angles between control points and endpoints in radians.
#     angle1 = math.atan2(beta[1], alpha[1])
#     angle2 = math.atan2(beta[2], alpha[2])

#     # Calculate the time coefficients, which gives information about how the
#     # curve's shape changes over time.
#     time_range = np.max(stroke.time_stamps) - np.min(stroke.time_stamps)
#     time_coefficients = np.array(
#         [time_range * alpha[1], time_range * alpha[2], time_range * alpha[3]]
#     )

#     # Determine if the stroke is a pen-up or pen-down curve.
#     pen_up_flag = stroke.pen_ups[-1]  # 1 if pen-up, 0 if pen-down.

#     return np.array(
#         [
#             *end_point_diff,
#             control_point_dist1,
#             control_point_dist2,
#             angle1,
#             angle2,
#             *time_coefficients,
#             pen_up_flag,
#         ]
#     ).reshape(
#         1, -1
#     )  # Will be a 2D array with shape (1, 10).


def _filter_and_get_stroke_file_paths(
    *, root: str, stroke_files: list[str]
) -> list[Path]:
    """Filter the stroke files and return a list of pathlib.Path objects representing the files.

    Args:
        root (str): The root directory of the stroke files.
        stroke_files (list[str]): A list of file names of the stroke files.

    Returns:
        list[Path]: A list of pathlib.Path objects representing the stroke files.
    """
    return sorted(
        # Get the full path to the stroke file.
        Path(root) / Path(file)
        for file in stroke_files
        if file.endswith(".xml")
        # Don't include these files. They are invalid or missing data.
        and not file.startswith(
            "z01-000z"
        )  # Exclude files starting with "z01-000z".
        and not file.startswith(
            "a08-551z-08"
        )  # Exclude files starting with "a08-551z-08".
        and not file.startswith(
            "a08-551z-09"
        )  # Exclude files starting with "a08-551z-09".
    )


def _set_up_train_val_test_data_stores() -> (
    tuple[BezierData, BezierData, BezierData, BezierData]
):
    """Set up the data stores for the train, first validation, second validation, and test sets.

    Returns:
        tuple[BezierData, BezierData, BezierData, BezierData]: A tuple of BezierData objects for
            the train, first validation, second validation, and test sets.

    Raises:
        ValueError: If the train, first validation, second validation, and test sets are not
            disjoint.
    """

    def get_train_val_test_label_file_names() -> (
        tuple[set[str], set[str], set[str], set[str]]
    ):
        """Get the label file names for train, first validation, second validation, and test sets.

        Returns:
            tuple[set[str], set[str], set[str], set[str]]: A tuple of sets of label file names for
                train, first validation, second validation, and test sets.
        """

        def _get_dataset_label_file_names(
            dataset_type: DatasetType,
        ) -> set[str]:
            """Get the label file names for the given dataset.

            Args:
                dataset_type (DatasetType): The dataset type.

            Returns:
                set[str]: A set of label file names for the given dataset.
            """
            label_file_names = set()
            with open(
                dataset_type.value, "r", encoding="utf-8"
            ) as label_files:
                for label_file_name in label_files:
                    label_file_names.add(label_file_name.strip())
            return label_file_names

        train_data_file_names = _get_dataset_label_file_names(
            DatasetType.TRAIN
        )
        val_1_data_file_names = _get_dataset_label_file_names(
            DatasetType.VAL_1
        )
        val_2_data_file_names = _get_dataset_label_file_names(
            DatasetType.VAL_2
        )
        test_data_file_names = _get_dataset_label_file_names(DatasetType.TEST)

        # Check that the train, first validation, second validation, and test sets are disjoint.
        if not train_data_file_names.isdisjoint(val_1_data_file_names):
            raise ValueError(
                "The train and first validation sets are not disjoint."
            )
        if not train_data_file_names.isdisjoint(val_2_data_file_names):
            raise ValueError(
                "The train and second validation sets are not disjoint."
            )
        if not train_data_file_names.isdisjoint(test_data_file_names):
            raise ValueError("The train and test sets are not disjoint.")
        if not val_1_data_file_names.isdisjoint(val_2_data_file_names):
            raise ValueError(
                "The first and second validation sets are not disjoint."
            )
        if not val_1_data_file_names.isdisjoint(test_data_file_names):
            raise ValueError(
                "The first validation and test sets are not disjoint."
            )
        if not val_2_data_file_names.isdisjoint(test_data_file_names):
            raise ValueError(
                "The second validation and test sets are not disjoint."
            )

        return (
            train_data_file_names,
            val_1_data_file_names,
            val_2_data_file_names,
            test_data_file_names,
        )

    (
        train_label_file_names,
        val_1_label_file_names,
        val_2_label_file_names,
        test_label_file_names,
    ) = get_train_val_test_label_file_names()

    train_data = BezierData(
        label_file_names=train_label_file_names
    )  # The train dataset.

    val_1_data = BezierData(
        label_file_names=val_1_label_file_names
    )  # The first validation dataset.

    val_2_data = BezierData(
        label_file_names=val_2_label_file_names
    )  # The second validation dataset.

    test_data = BezierData(
        label_file_names=test_label_file_names
    )  # The test dataset.

    return train_data, val_1_data, val_2_data, test_data


def _append_label_bezier_curves_data(
    *,
    label: str,
    labels_file_name: str,
    bezier_curves_data: list[npt.NDArray[np.float_]],
    train_data: BezierData,
    val_1_data: BezierData,
    val_2_data: BezierData,
    test_data: BezierData,
) -> None:
    """Append the label and Bezier curves data to the appropriate dataset.

    The data will not be appended if the label file name is not in any of the datasets.

    Args:
        label (str): The label of the stroke file.
        labels_file_name (str): The name of the labels file.
        bezier_curves_data (list[npt.NDArray[np.float_]]): A list of 2D numpy arrays containing
            Bezier curve information for each stroke in the stroke file.
        train_data (BezierData): The train dataset.
        val_1_data (BezierData): The first validation dataset.
        val_2_data (BezierData): The second validation dataset.
        test_data (BezierData): The test dataset.

    Returns:
        None.
    """
    if labels_file_name in train_data.label_file_names:
        train_data.labels.append(label)
        train_data.bezier_curves_data.append(bezier_curves_data)
    elif labels_file_name in val_1_data.label_file_names:
        val_1_data.labels.append(label)
        val_1_data.bezier_curves_data.append(bezier_curves_data)
    elif labels_file_name in val_2_data.label_file_names:
        val_2_data.labels.append(label)
        val_2_data.bezier_curves_data.append(bezier_curves_data)
    elif labels_file_name in test_data.label_file_names:
        test_data.labels.append(label)
        test_data.bezier_curves_data.append(bezier_curves_data)


def _convert_to_numpy_and_save(
    *,
    train_data: BezierData,
    val_1_data: BezierData,
    val_2_data: BezierData,
    test_data: BezierData,
):
    """Convert the data to numpy arrays.

    Args:
        train_data (BezierData): The train data.
        val_1_data (BezierData): The first validation data.
        val_2_data (BezierData): The second validation data.
        test_data (BezierData): The test data.

    Raises:
        ValueError: If the data is invalid.
    """

    def is_valid_bezier_data(data: BezierData) -> None:
        """Check that the data is valid.

        Args:
            data (BezierData): The data to check.

        Raises:
            ValueError: If the data is invalid.
        """
        if len(data.labels) != len(data.bezier_curves_data):
            raise ValueError(
                f"Length of labels ({len(data.labels)}) does not match length of Bezier curves "
                f"data ({len(data.bezier_curves_data)})."
            )

    def pad_bezier_curves_data(
        all_bezier_curves_data: list[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Pad the Bezier curves data with -1 so that all strokes have the same number of curves.

        Args:
            all_bezier_curves_data (list[npt.NDArray[np.float_]]): A list of 2D numpy arrays
                containing Bezier curve information for each stroke. Each array has shape
                (number_of_curves, 10), with 10 columns representing various
                properties of the Bezier curve.

        Returns:
            npt.NDArray[np.float_]: A 3D numpy array with shape
                (number_of_strokes, max_number_of_curves, 10), where each row represents the Bezier
                curve information of a stroke, and each stroke is padded with -1 values to have the
                same number of curves.

        Raises:
            ValueError: If there is no Bezier curve data.
        """
        if not all_bezier_curves_data:
            raise ValueError("There is no Bezier curve data.")

        max_num_curves = max(len(curves) for curves in all_bezier_curves_data)

        return np.array(
            [
                np.concatenate(
                    (
                        curves,
                        np.full((max_num_curves - len(curves), 1, 10), -1),
                    ),
                    axis=0,
                )
                for curves in all_bezier_curves_data
            ]
        )

    def convert_to_numpy(
        data: BezierData,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Convert the data to numpy arrays.

        Args:
            data (BezierData): The data to convert.

        Returns:
            tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]: A tuple containing the labels
                and Bezier curves data as numpy arrays.

        Raises:
            AssertionError: If the number of labels does not match the number of Bezier curves.
        """
        labels_arr = np.array(data.labels)
        bezier_curves_arr = pad_bezier_curves_data(data.bezier_curves_data)

        assert labels_arr.shape[0] == bezier_curves_arr.shape[0], (
            f"Number of labels ({labels_arr.shape[0]}) does not match number of Bezier curves "
            f"({bezier_curves_arr.shape[0]})."
        )

        return labels_arr, bezier_curves_arr

    # Check that the data is valid.
    for data in [train_data, val_1_data, val_2_data, test_data]:
        is_valid_bezier_data(data)

    # Convert the data to numpy arrays.
    train_labels, train_bezier_curves = convert_to_numpy(train_data)
    val_1_labels, val_1_bezier_curves = convert_to_numpy(val_1_data)
    val_2_labels, val_2_bezier_curves = convert_to_numpy(val_2_data)
    test_labels, test_bezier_curves = convert_to_numpy(test_data)

    # Create a data directory if it doesn't exist.
    Path("data").mkdir(parents=True, exist_ok=True)

    # Save the data to a numpy .npz file.
    np.savez_compressed(
        EXTRACTED_DATA_PATH,
        train_labels=train_labels,
        train_bezier_curves=train_bezier_curves,
        val_1_labels=val_1_labels,
        val_1_bezier_curves=val_1_bezier_curves,
        val_2_labels=val_2_labels,
        val_2_bezier_curves=val_2_bezier_curves,
        test_labels=test_labels,
        test_bezier_curves=test_bezier_curves,
    )


def extract_all_data() -> None:
    """Extract all data from the IAM On-Line Handwriting Database and save it to a numpy .npz file.

    This function processes the data and saves it in the following format:
    - labels: a list of strings, where each string represents a line label from the database.

    - bezier_data: a 2D numpy array with shape (number_of_strokes, 10), where each row
        represents the Bezier curve information of a stroke. The columns contain the following:
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
    (
        train_data,
        val_1_data,
        val_2_data,
        test_data,
    ) = _set_up_train_val_test_data_stores()

    # Get the total number of directories to process.
    total_dirs = sum(1 for _ in os.walk(LINE_STROKES_DATA_DIR))

    # Go through each directory in the line strokes data directory.
    for root, _, stroke_files in tqdm(
        os.walk(LINE_STROKES_DATA_DIR),
        total=total_dirs,
        desc="Processing directories",
    ):
        # Get all stroke files in the directory.
        stroke_files = _filter_and_get_stroke_file_paths(
            root=root, stroke_files=stroke_files
        )

        # Go through each stroke file in the directory.
        for stroke_file in stroke_files:
            # Get the label for the stroke file.
            line_label, labels_file_name = _get_label_from_stroke_file(
                stroke_file
            )
            if line_label is None:
                break

            # Get the strokes from the stroke file.
            strokes = _get_strokes_from_stroke_file(stroke_file)

            # Compute the Bezier curves for the stroke file.
            bezier_curves_data = [
                _fit_stroke_with_bezier_curve(stroke) for stroke in strokes
            ]

            # Append the label and Bezier curve data to the appropriate dataset.
            _append_label_bezier_curves_data(
                label=line_label,
                labels_file_name=labels_file_name,
                bezier_curves_data=bezier_curves_data,
                train_data=train_data,
                val_1_data=val_1_data,
                val_2_data=val_2_data,
                test_data=test_data,
            )

    # Convert the data to numpy arrays and save it to a .npz file.
    _convert_to_numpy_and_save(
        train_data=train_data,
        val_1_data=val_1_data,
        val_2_data=val_2_data,
        test_data=test_data,
    )
    