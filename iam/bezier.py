"""A module for Bezier curve functions. Used to convert stroke data to Bezier curves."""
from dataclasses import dataclass, field
import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# import scipy.optimize
from scipy.special import comb


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


def _get_bezier_control_points(
    x_points: list[float], y_points: list[float], degree: int = 3
) -> list[list[float]]:
    """Compute least square bezier curve fit using penrose pseudoinverse.

    Args:
        x_points (list[float]): X-coordinates of the points to fit.
        y_points (list[float]): Y-coordinates of the points to fit.
        degree (int, optional): Degree of the Bezier curve. Defaults to 3.

    Returns:
        list[list[float]]: A list of control points for the Bezier curve (e.g. [[x0, y0], [x1, y1]])

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
        """Compute the Bernstein polynomial when a = 0 and b = 1.

        Args:
            n (int): The degree of the polynomial.
            t (float): The variable of the polynomial.
            k (int): The current term of the polynomial.

        Returns:
            float: The value of the Bernstein polynomial.
        """
        return t**k * (1 - t) ** (n - k) * comb(n, k)

    def bernstein_matrix(T: npt.NDArray[np.float_]) -> np.matrix:
        """Compute the Bernstein matrix for Bezier curves.

        Args:
            T (npt.NDArray[np.float64]): The range of the variable t for the Bernstein polynomial.

        Returns:
            np.matrix: The Bernstein matrix.
        """
        return np.matrix(
            [
                [bernstein_poly(degree, t, k) for k in range(degree + 1)]
                for t in T
            ]
        )

    def least_square_fit(
        points: npt.NDArray[np.float_], M: np.matrix
    ) -> np.matrix:
        """Perform least square fit using pseudoinverse.

        Args:
            points (npt.NDArray[np.float64]): The points to fit.
            M (np.matrix): The Bernstein matrix.

        Returns:
            np.matrix: The Bezier parameters.
        """
        M_pseudoinv = np.linalg.pinv(M)
        return M_pseudoinv * points

    T = np.linspace(0, 1, len(x_points))
    M = bernstein_matrix(T)
    points = np.array(list(zip(x_points, y_points)))

    control_points = least_square_fit(points, M).tolist()
    control_points[0] = [x_points[0], y_points[0]]
    control_points[-1] = [x_points[-1], y_points[-1]]

    return control_points


def _bernstein_poly(i: int, n: int, t: np.float64) -> np.float64:
    """The Bernstein polynomial of n, i as a function of t.

    Args:
        i (int): The current term of the polynomial.
        n (int): The degree of the polynomial.
        t (np.float64): The variable of the polynomial.

    Returns:
        np.float64: The value of the Bernstein polynomial.
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def _bezier_curve(
    control_points: list[list[float]], num_time_steps: int = 50
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Given a set of control points, return the bezier curve defined by the control points.

    Args:
        control_points (list[list[float]]): List of control points where each point
            is a list of two floats [x, y] such as [[1, 1], [2, 8], [3, 15], ... , [Xn, Yn]].
        num_time_steps (int): The number of time steps, defaults to 50.

    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Tuple containing two numpy arrays
            for x and y coordinates of the points on the bezier curve.
    """
    num_points = len(control_points)
    x_points = np.array([point[0] for point in control_points])
    y_points = np.array([point[1] for point in control_points])

    t_values = np.linspace(0.0, 1.0, num_time_steps)

    polynomial_array = np.array(
        [
            _bernstein_poly(i, num_points - 1, t_values)
            for i in range(num_points)
        ]
    )

    x_vals = np.dot(x_points, polynomial_array)
    y_vals = np.dot(y_points, polynomial_array)

    return x_vals, y_vals


def _plot_stroke_and_bezier_curve(
    stroke_data: StrokeData, bezier_control_points: npt.NDArray[np.float_]
) -> None:
    """Plots the stroke data and the Bezier curve.

    Args:
        stroke_data (StrokeData): The stroke data.
        bezier_control_points (npt.NDArray[np.float_]): The Bezier control points.

    Returns:
        None: The plot is shown.
    """
    x_points = stroke_data.x_points
    y_points = stroke_data.y_points

    control_x_points = [point[0] for point in bezier_control_points]
    control_y_points = [point[1] for point in bezier_control_points]

    plt.plot(x_points, y_points, "ro", label="Stroke Points")
    plt.plot(
        control_x_points, control_y_points, "k--o", label="Control Points"
    )

    bezier_x_points, bezier_y_points = _bezier_curve(
        bezier_control_points, num_time_steps=1000
    )
    plt.plot(bezier_x_points, bezier_y_points, "b-", label="Bezier Curve")

    plt.legend()
    plt.show()


def fit_stroke_with_bezier_curve(
    *, stroke: StrokeData, degree: int
) -> npt.NDArray[np.float_]:
    """Processes the stroke data and fits it with a Bezier curve.

    Args:
        stroke (StrokeData): The stroke data.
        degree (int): The degree of the Bezier curve.

    Returns:
        A numpy array containing the calculated Bezier curve information.

    Raises:
        ValueError: If the stroke data is invalid.
        ValueError: If the degree is invalid.
    """
    if not isinstance(stroke, StrokeData):
        raise ValueError(
            f"Invalid stroke data: {stroke}. Must be an instance of {StrokeData}."
        )
    if not isinstance(degree, int):
        raise ValueError(f"Invalid degree: {degree}. Must be an integer.")

    # Check if the stroke data has enough points.
    if len(stroke.x_points) < degree + 1:
        return np.zeros((1, 10))

    # Get the Bezier control points for the stroke data.
    bezier_control_points = _get_bezier_control_points(
        stroke.x_points, stroke.y_points, degree=degree
    )

    # Plot the stroke data and the Bezier curve.
    # _plot_stroke_and_bezier_curve(stroke, bezier_control_points)

    # Compute the distance between end points
    end_point_diff = np.array(
        [
            bezier_control_points[-1][0] - bezier_control_points[0][0],
            bezier_control_points[-1][1] - bezier_control_points[0][1],
        ]
    )
    end_point_diff_norm = max(np.linalg.norm(end_point_diff), 0.001)

    # Compute the distance between the first control point and the starting point
    control_point_dist1 = (
        np.linalg.norm(
            np.array(
                [
                    bezier_control_points[1][0] - bezier_control_points[0][0],
                    bezier_control_points[1][1] - bezier_control_points[0][1],
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
                    bezier_control_points[2][0] - bezier_control_points[0][0],
                    bezier_control_points[2][1] - bezier_control_points[0][1],
                ]
            )
        )
        / end_point_diff_norm
    )

    # Get the angles between control points and end points in radians.
    angle1 = math.atan2(
        bezier_control_points[1][1] - bezier_control_points[0][1],
        bezier_control_points[1][0] - bezier_control_points[0][0],
    )
    angle2 = math.atan2(
        bezier_control_points[2][1] - bezier_control_points[0][1],
        bezier_control_points[2][0] - bezier_control_points[0][0],
    )

    # Calculate the time coefficients
    time_range = np.max(stroke.time_stamps) - np.min(stroke.time_stamps)
    time_coefficients = np.array(
        [
            time_range * bezier_control_points[1][0],
            time_range * bezier_control_points[2][0],
            time_range * bezier_control_points[3][0],
        ]
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


# Slower version of the above function, using scipy.optimize.minimize.

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
