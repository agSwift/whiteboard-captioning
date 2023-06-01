"""Used to extract data from the IAM On-Line Handwriting Database.

The database can be downloaded from:
https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database.

The following files were filtered out when extracting the data, as they are invalid or missing data:
a08-551z-08.xml, a08-551z-09.xml, and all z01-000z stroke XML files.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

import os
import xml.etree.ElementTree as ET
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from iam import bezier_curves

LINE_STROKES_DATA_DIR = Path("backend/datasets/IAM/lineStrokes")
LINE_LABELS_DATA_DIR = Path("backend/datasets/IAM/ascii")
LINE_IMAGES_DATA_DIR = Path("backend/datasets/IAM/lineImages")

EXTRACTED_DATA_TEMPLATE = "iam_data_{validation}_val_degree_{degree}"


class DatasetType(Enum):
    """An enum for the dataset types."""

    TRAIN_CROSS_VAL = Path(
        "backend/datasets/IAM/trainset.txt"
    )  # Used during cross validation.
    VAL_1 = Path("backend/datasets/IAM/valset1.txt")
    VAL_2 = Path("backend/datasets/IAM/valset2.txt")
    TEST = Path("backend/datasets/IAM/testset.txt")
    TRAIN_SINGLE_VAL = None  # Train and val_1 combined. Used during validation on single dataset.


def get_extracted_data_file_path(
    *, with_cross_val: bool, bezier_degree: int
) -> Path:
    """Gets the extracted data file path.

    Args:
        with_cross_val (bool): Whether cross validation is used.
        bezier_degree (int): The bezier curve degree.

    Returns:
        The extracted data file path.
    """
    if not isinstance(with_cross_val, bool):
        raise ValueError(
            f"Invalid cross validation value: {with_cross_val}. Must be a boolean."
        )
    if not isinstance(bezier_degree, int):
        raise ValueError(
            f"Invalid bezier degree value: {bezier_degree}. Must be an integer."
        )

    extracted_data_file_name = EXTRACTED_DATA_TEMPLATE.format(
        validation="cross" if with_cross_val else "single",
        degree=bezier_degree,
    )

    return Path(f"backend/iam/data/{extracted_data_file_name}.npz")


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
) -> tuple[Optional[str], str, Path]:
    """Gets the line label and the labels file name that the label belongs to.

    Args:
        stroke_file (Path): The stroke file path.

    Returns:
        The line label, the label file name, and the image file path.
        None if the line label is not found.

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
    line_label_idx_chars = stroke_file_name[-2:]  # e.g. 01
    if not line_label_idx_chars.isdigit():
        raise ValueError(
            f"Invalid stroke file: {stroke_file}. "
            f"Label line index must be a number."
        )
    line_label_idx = int(line_label_idx_chars)

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

    image_file = (
        LINE_IMAGES_DATA_DIR
        / root_labels_dir
        / sub_labels_dir
        / f"{labels_file_name}-{line_label_idx_chars}.tif"
    )
    assert image_file.exists(), f"Image file not found: {image_file}."

    # Get the line label from the labels file.
    return (
        _get_line_from_labels_file(labels_file, line_label_idx),
        labels_file_name,
        image_file,
    )


def _parse_stroke_element(stroke_elem: ET.Element) -> bezier_curves.StrokeData:
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

    return bezier_curves.StrokeData(x_points, y_points, time_stamps, pen_ups)


def _get_strokes_from_stroke_file(
    stroke_file: Path,
) -> list[bezier_curves.StrokeData]:
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
    # max_y = float(
    #     root.find("WhiteboardDescription/DiagonallyOppositeCoords").attrib["y"]
    # )
    # max_x = float(
    #     root.find("WhiteboardDescription/DiagonallyOppositeCoords").attrib["x"]
    # )

    for stroke in strokes:
        max_x = max(stroke.x_points)
        max_y = max(stroke.y_points)
        min_x = min(stroke.x_points)
        min_y = min(stroke.y_points)

        # Shift x and y points to start at 0.
        stroke.x_points = [x - min_x for x in stroke.x_points]
        stroke.y_points = [y - min_y for y in stroke.y_points]

        # Flip the points in the y-axis.
        # stroke.y_points = [(max_y - min_y) - y for y in stroke.y_points]

        # Scale so that the x and y points are between 0 and 1.
        scale_y = (
            1 if max_y - min_y == 0 else 1 / (max_y - min_y)
        )  # Avoid division by 0.

        scale_x = (
            1 if max_x - min_x == 0 else 1 / (max_x - min_x)
        )  # Avoid division by 0.

        stroke.x_points = [x * scale_x for x in stroke.x_points]
        stroke.y_points = [y * scale_y for y in stroke.y_points]

        assert all(0 <= y <= 1 for y in stroke.y_points), (
            f"Invalid stroke file: {stroke_file}. "
            f"All y-points must be between 0 and 1."
        )

        assert all(0 <= x <= 1 for x in stroke.x_points), (
            f"Invalid stroke file: {stroke_file}. "
            f"All x-points must be between 0 and 1."
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
    stroke_file_paths = [
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
    ]

    assert all(
        not (
            file.name.startswith("z01-000z")
            and file.name.startswith("a08-551z-08")
            and file.name.startswith("a08-551z-09")
        )
        for file in stroke_file_paths
    ), (
        "Invalid stroke files. "
        'The stroke files must not start with "z01-000z", "a08-551z-08", or "a08-551z-09".'
    )

    return stroke_file_paths


def _set_up_train_val_test_data_stores() -> (
    tuple[
        bezier_curves.BezierData,
        bezier_curves.BezierData,
        bezier_curves.BezierData,
        bezier_curves.BezierData,
    ]
):
    """Set up the data stores for the train, first validation, second validation, and test sets.

    Returns:
        tuple[bezier_curves.BezierData, bezier_curves.BezierData, bezier_curves.BezierData, bezier_curves.BezierData]:
            A tuple of bezier_curves.BezierData objects for the train,
            first validation, second validation, and test sets.

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

        train_cross_val_data_file_names = _get_dataset_label_file_names(
            DatasetType.TRAIN_CROSS_VAL
        )
        val_1_data_file_names = _get_dataset_label_file_names(
            DatasetType.VAL_1
        )
        val_2_data_file_names = _get_dataset_label_file_names(
            DatasetType.VAL_2
        )
        test_data_file_names = _get_dataset_label_file_names(DatasetType.TEST)

        # Check that the train, first validation, second validation, and test sets are disjoint.
        if not train_cross_val_data_file_names.isdisjoint(
            val_1_data_file_names
        ):
            raise ValueError(
                "The train and first validation sets are not disjoint."
            )
        if not train_cross_val_data_file_names.isdisjoint(
            val_2_data_file_names
        ):
            raise ValueError(
                "The train and second validation sets are not disjoint."
            )
        if not train_cross_val_data_file_names.isdisjoint(
            test_data_file_names
        ):
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
            train_cross_val_data_file_names,
            val_1_data_file_names,
            val_2_data_file_names,
            test_data_file_names,
        )

    (
        train_cross_val_label_file_names,
        val_1_label_file_names,
        val_2_label_file_names,
        test_label_file_names,
    ) = get_train_val_test_label_file_names()

    train_cross_val_data = bezier_curves.BezierData(
        label_file_names=train_cross_val_label_file_names
    )  # The train dataset used for cross validation.

    val_1_data = bezier_curves.BezierData(
        label_file_names=val_1_label_file_names
    )  # The first validation dataset.

    val_2_data = bezier_curves.BezierData(
        label_file_names=val_2_label_file_names
    )  # The second validation dataset.

    test_data = bezier_curves.BezierData(
        label_file_names=test_label_file_names
    )  # The test dataset.

    return train_cross_val_data, val_1_data, val_2_data, test_data


def _convert_to_numpy_and_save(
    *,
    train_cross_val_data: bezier_curves.BezierData,
    val_1_data: bezier_curves.BezierData,
    val_2_data: bezier_curves.BezierData,
    test_data: bezier_curves.BezierData,
    with_cross_val: bool,
    bezier_degree: int,
) -> None:
    """Convert the data to numpy arrays.

    A larger training dataset (train_cross_val_data combined with
    val_1_data) is also created and saved.

    Args:
        train_cross_val_data (bezier_curves.BezierData): The train data used for cross validation.
        val_1_data (bezier_curves.BezierData): The first validation data.
        val_2_data (bezier_curves.BezierData): The second validation data.
        test_data (bezier_curves.BezierData): The test data.
        with_cross_val (bool): Whether or not cross validation is being used.
        bezier_degree (int): The degree of the Bezier curves.

    Returns:
        None.

    Raises:
        ValueError: If the data is invalid.
    """

    def convert_to_numpy(
        data: bezier_curves.BezierData,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Convert the data to numpy arrays.

        Args:
            data (bezier_curves.BezierData): The data to convert.

        Returns:
            tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]: A tuple containing the labels
                and Bezier curves data as numpy arrays.

        Raises:
            AssertionError: If the number of labels does not match the number of Bezier curves.
        """
        labels_arr = np.array(data.labels)
        bezier_curves_arr = bezier_curves.pad_bezier_curves_data(
            data.bezier_curves_data
        )

        assert labels_arr.shape[0] == bezier_curves_arr.shape[0], (
            f"Number of labels ({labels_arr.shape[0]}) does not match number of Bezier curves "
            f"({bezier_curves_arr.shape[0]})."
        )

        return labels_arr, bezier_curves_arr

    # Check that the data is valid.
    for data in [train_cross_val_data, val_1_data, val_2_data, test_data]:
        bezier_curves.is_valid_bezier_data(data)

    # Convert the data to numpy arrays.
    train_cross_val_labels, train_cross_val_bezier_curves = convert_to_numpy(
        train_cross_val_data
    )
    val_1_labels, val_1_bezier_curves = convert_to_numpy(val_1_data)
    val_2_labels, val_2_bezier_curves = convert_to_numpy(val_2_data)
    test_labels, test_bezier_curves = convert_to_numpy(test_data)

    # Create a larger train set used during single validation by combining
    # the train_cross_val_data and val_1_data sets.
    num_train_larger_curves = max(
        train_cross_val_bezier_curves.shape[1], val_1_bezier_curves.shape[1]
    )
    num_bezier_curve_features = len(train_cross_val_bezier_curves[0][0][0])

    # Add further padding to the train_cross_val_bezier_curves
    # val_1_bezier_curves data, so that they are then same size
    # and can be concatenated.
    train_cross_val_bezier_curves_padded = np.concatenate(
        (
            train_cross_val_bezier_curves,
            np.full(
                (
                    train_cross_val_bezier_curves.shape[0],
                    num_train_larger_curves
                    - train_cross_val_bezier_curves.shape[1],
                    1,
                    num_bezier_curve_features,
                ),
                -1,
            ),
        ),
        axis=1,
    )
    val_1_bezier_curves_padded = np.concatenate(
        (
            val_1_bezier_curves,
            np.full(
                (
                    val_1_bezier_curves.shape[0],
                    num_train_larger_curves - val_1_bezier_curves.shape[1],
                    1,
                    num_bezier_curve_features,
                ),
                -1,
            ),
        ),
        axis=1,
    )

    train_single_val_labels = np.concatenate(
        (train_cross_val_labels, val_1_labels)
    )
    train_single_val_bezier_curves = np.concatenate(
        (train_cross_val_bezier_curves_padded, val_1_bezier_curves_padded)
    )

    # Create a data directory if it doesn't exist.
    Path("backend/iam/data").mkdir(parents=True, exist_ok=True)

    extracted_data_file_path = get_extracted_data_file_path(
        with_cross_val=with_cross_val, bezier_degree=bezier_degree
    )

    # Save the data to a numpy .npz file.
    np.savez_compressed(
        extracted_data_file_path,
        train_cross_val_labels=train_cross_val_labels,
        train_cross_val_bezier_curves=train_cross_val_bezier_curves,
        val_1_labels=val_1_labels,
        val_1_bezier_curves=val_1_bezier_curves,
        val_2_labels=val_2_labels,
        val_2_bezier_curves=val_2_bezier_curves,
        test_labels=test_labels,
        test_bezier_curves=test_bezier_curves,
        train_single_val_labels=train_single_val_labels,
        train_single_val_bezier_curves=train_single_val_bezier_curves,
    )


def extract_all_data(
    *,
    with_cross_val: bool,
    bezier_degree: int,
) -> None:
    """Extract all data from the IAM On-Line Handwriting Database and save it to a numpy .npz file.

    This function processes the data and saves it in the following format:
    - labels: a list of strings, where each string represents a line label from the database.

    - bezier_data: a 2D numpy array with shape (number_of_strokes, num_bezier_curve_features),
        where each row represents the Bezier curve information of a stroke. The columns
        contain the following:
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

    Args:
        with_cross_val: Whether cross validation is used.
        bezier_degree: The degree of the Bezier curves to use.

    Returns:
        None. The data is saved to a numpy .npz file.

    Raises:
        TypeError: If bezier_curve_degree is not an int.
    """
    if not isinstance(bezier_degree, int):
        raise TypeError(
            f"bezier_curve_degree must be an int, not {type(bezier_degree)}"
        )

    (
        train_cross_val_data,
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
            (
                line_label,
                labels_file_name,
                _,
            ) = _get_label_from_stroke_file(stroke_file)
            if line_label is None:
                break

            # Get the strokes from the stroke file.
            strokes = _get_strokes_from_stroke_file(stroke_file)

            # Compute the Bezier curves for the stroke file.
            bezier_curves_data = []
            for stroke in strokes:
                (
                    _,
                    _,
                    bezier_curve_features,
                ) = bezier_curves.fit_stroke_with_bezier_curve(
                    stroke=stroke, degree=bezier_degree
                )
                bezier_curves_data.append(bezier_curve_features)

            # Append the label and Bezier curve data to the appropriate dataset.
            bezier_curves.append_label_bezier_curves_data(
                label=line_label,
                labels_file_name=labels_file_name,
                bezier_curves_data=bezier_curves_data,
                train_cross_val_data=train_cross_val_data,
                val_1_data=val_1_data,
                val_2_data=val_2_data,
                test_data=test_data,
            )

    # Convert the data to numpy arrays and save it to a .npz file.
    # A larger training set is also created and saved in this step.
    _convert_to_numpy_and_save(
        train_cross_val_data=train_cross_val_data,
        val_1_data=val_1_data,
        val_2_data=val_2_data,
        test_data=test_data,
        bezier_degree=bezier_degree,
        with_cross_val=with_cross_val,
    )
