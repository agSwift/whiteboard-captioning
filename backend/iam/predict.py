"""Used for loading trained models and predicting the characters in a stroke."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from iam import bezier_curves, dataset, extraction, train

# ALL_NUM_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# ALL_BEZIER_CURVE_DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# ALL_MODEL_TYPES = list(train.ModelType)
ALL_NUM_LAYERS = [3]
ALL_BEZIER_CURVE_DEGREES = [5]
ALL_MODEL_TYPES = [train.ModelType.LSTM]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _load_models():
    loaded_models = {}

    for model_type in ALL_MODEL_TYPES:
        for bezier_curve_degree in ALL_BEZIER_CURVE_DEGREES:
            for num_layers in ALL_NUM_LAYERS:
                for bidirectional in [True]:
                    # Create the model file name.
                    model_file_name = train.get_model_file_name(
                        model_name=model_type.name.lower(),
                        bezier_curve_degree=bezier_curve_degree,
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                    )

                    # Ensure the model exists.
                    model_path = Path(
                        f"backend/iam/models/{model_file_name}.ckpt"
                    )
                    assert model_path.exists(), (
                        f"Model {model_file_name} does not exist.\n"
                        "Run train.py to train the model."
                    )

                    # Get the bezier curve dimension for the model.
                    extracted_data_path = (
                        extraction.get_extracted_data_file_path(
                            with_cross_val=train.CROSS_VALIDATION,
                            bezier_degree=bezier_curve_degree,
                        )
                    )
                    assert (
                        extracted_data_path.exists()
                    ), f"Extracted data {extracted_data_path} does not exist.\n"
                    all_bezier_data = np.load(extracted_data_path)

                    train_cross_val_dataset = dataset.StrokeBezierDataset(
                        all_bezier_data=all_bezier_data,
                        dataset_type=extraction.DatasetType.TRAIN_CROSS_VAL,
                    )
                    bezier_curve_dimension = (
                        train_cross_val_dataset.all_bezier_curves[0].shape[-1]
                    )

                    # Load the model.
                    model = model_type.value(
                        bezier_curve_dimension=bezier_curve_dimension,
                        hidden_size=train.HIDDEN_SIZE,
                        num_classes=train.NUM_CLASSES,
                        num_layers=num_layers,
                        dropout=train.DROPOUT_RATE,
                        bidirectional=bidirectional,
                        device=DEVICE,
                    ).to(DEVICE)
                    model.load_state_dict(
                        torch.load(model_path, map_location=DEVICE)
                    )

                    # Add the model to the dictionary.
                    loaded_models[model_file_name] = model

    return loaded_models


LOADED_MODELS = _load_models()


def _get_downsampled_stroke_data(
    *,
    x_points: list[float],
    y_points: list[float],
    time_stamps: list[float],
    bezier_curve_degree: int,
    points_per_second: int,
) -> bezier_curves.StrokeData:
    """Downsamples the stroke data to the given number of points per second.

    Args:
        x_points (list[float]): The x points of the stroke.
        y_points (list[float]): The y points of the stroke.
        time_stamps (list[float]): The time stamps of the stroke.
        bezier_curve_degree (int): The degree of the bezier curves to fit to the strokes.
        points_per_second (int): The number of points to keep per second.

    Returns:
        bezier_curves.StrokeData: A bezier_curves.StrokeData object.

    Raises:
        AssertionError: If the number of downsampled x points, y points, time stamps,
            and pen ups are not equal.
    """
    x_points_sampled = []
    y_points_sampled = []
    time_stamps_sampled = []
    pen_ups_sampled = []

    total_time = time_stamps[-1] - time_stamps[0]
    total_points = int(total_time * points_per_second)

    # Ensure total_points is at least bezier_curve_degree + 1.
    if total_points < bezier_curve_degree + 1:
        total_points = len(time_stamps)
    else:
        # Ensure total points is a multiple of bezier_curve_degree + 1.
        total_points -= total_points % (bezier_curve_degree + 1)

    indices_to_keep = np.linspace(
        0, len(time_stamps) - 1, total_points, dtype=int
    )

    for i in indices_to_keep:
        x_points_sampled.append(x_points[i])
        y_points_sampled.append(y_points[i])
        time_stamps_sampled.append(time_stamps[i])

        if i == indices_to_keep[-1]:
            # If this is the last point in the stroke, the pen is up.
            pen_ups_sampled.append(1)
        else:
            # The pen is down.
            pen_ups_sampled.append(0)

    assert (
        len(x_points_sampled)
        == len(y_points_sampled)
        == len(time_stamps_sampled)
        == len(pen_ups_sampled)
    ), (
        f"Number of x points ({len(x_points_sampled)}), y points ({len(y_points_sampled)}), "
        f"time stamps ({len(time_stamps_sampled)}), and pen ups ({len(pen_ups_sampled)}) "
        "are not equal."
    )

    return bezier_curves.StrokeData(
        x_points=x_points_sampled,
        y_points=y_points_sampled,
        time_stamps=time_stamps_sampled,
        pen_ups=pen_ups_sampled,
    )


def greedy_predict(
    strokes: list[list[dict[str, float]]],
    model_name: str,
    bezier_curve_degree: int,
    num_layers: int,
    bidirectional: bool,
    points_per_second: int,
) -> tuple[str, list[float], list[float], list[float], list[float]]:
    """Predict the characters in the strokes.

    Uses greedy decoding to convert the logits to a string.

    Args:
        strokes (list[list[dict[str, float]]]): A list of strokes. Each stroke is a list of points.
            Each point is a dictionary with keys "x", "y", and "time".
        model_name (str): The name of the model to use.
        bezier_curve_degree (int): The degree of the bezier curves to fit to the strokes.
        num_layers (int): The number of recurrent layers in the model.
        bidirectional (bool): Whether the model is bidirectional.
        points_per_second (int): The number of points to keep per second.

    Returns:
        str, list[float], list[float], list[float], list[float]: A tuple containing the predicted
            string and the x points and y points of the stroke points and bezier curves.

    Raises:
        ValueError: If the model name is invalid.
        AssertionError: If the number of x points, y points, time stamps, and pen ups are not equal.
        AssertionError: If any y point is not between 0 and 1.
        AssertionError: If the model does not exist.
    """
    if model_name not in [model_type.name for model_type in train.ModelType]:
        raise ValueError(
            f"Invalid model name: {model_name}\n"
            f"Valid model names: {[model_type.name for model_type in train.ModelType]}"
        )

    # Parse the strokes into bezier_curves.StrokeData objects.
    all_stroke_data = []

    for stroke in strokes:
        # Get max and min x and y values.
        max_x = max(point["x"] for point in stroke)
        max_y = max(point["y"] for point in stroke)
        min_x = min(point["x"] for point in stroke)
        min_y = min(point["y"] for point in stroke)

        # Shift x and y points to start at 0.
        x_points_shifted = [point["x"] - min_x for point in stroke]
        y_points_shifted = [point["y"] - min_y for point in stroke]

        # Flip the y points so that the strokes are oriented correctly.
        y_points_shifted = [(max_y - min_y) - y for y in y_points_shifted]

        # Flip the points in the x-axis.
        # x_points_shifted = [(max_x - min_x) - x for x in x_points_shifted]

        # Scale so that the x and y points are between 0 and 1.
        scale_y = (
            1 if max_y - min_y == 0 else 1 / (max_y - min_y)
        )  # Avoid division by 0.

        scale_x = (
            1 if max_x - min_x == 0 else 1 / (max_x - min_x)
        )  # Avoid division by 0.

        x_points_scaled = [x * scale_x for x in x_points_shifted]
        y_points_scaled = [y * scale_y for y in y_points_shifted]
        time_stamps = [point["time"] for point in stroke]

        stroke_data = _get_downsampled_stroke_data(
            x_points=x_points_scaled[:],
            y_points=y_points_scaled[:],
            time_stamps=time_stamps[:],
            bezier_curve_degree=bezier_curve_degree,
            points_per_second=points_per_second,
        )

        assert all(0 <= y <= 1 for y in stroke_data.y_points), (
            f"Invalid stroke: {stroke}.\n"
            f"All y-points must be between 0 and 1 after downsampling."
        )

        assert all(0 <= x <= 1 for x in stroke_data.x_points), (
            f"Invalid stroke: {stroke}.\n"
            f"All x-points must be between 0 and 1 after downsampling."
        )

        all_stroke_data.append(stroke_data)

    assert len(all_stroke_data) == len(
        strokes
    ), "Number of strokes is not equal."

    # Fit bezier curves to the strokes.
    bezier_curves_data = []
    all_bezier_x_points = []
    all_bezier_y_points = []

    for stroke_data in all_stroke_data:
        (
            bezier_x_points,
            bezier_y_points,
            bezier_curve_features,
        ) = bezier_curves.fit_stroke_with_bezier_curve(
            stroke=stroke_data, degree=bezier_curve_degree
        )
        bezier_curves_data.append(bezier_curve_features)
        all_bezier_x_points.append(list(bezier_x_points))
        all_bezier_y_points.append(list(bezier_y_points))

    bezier_curves_data = np.array(bezier_curves_data)

    # Convert the bezier curves data to a tensor.
    bezier_curves_data = torch.tensor(bezier_curves_data, dtype=torch.float32)
    bezier_curves_data = bezier_curves_data.permute(1, 0, 2).to(DEVICE)

    # Load the required pytorch model.
    trained_model_file_name = train.get_model_file_name(
        model_name=model_name.lower(),
        bezier_curve_degree=bezier_curve_degree,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    assert trained_model_file_name in LOADED_MODELS, (
        f"Model {trained_model_file_name} does not exist.\n"
        "Please run train.py to train the model."
    )
    model = LOADED_MODELS[trained_model_file_name]

    # Predict the characters.
    logits = model(bezier_curves_data)

    # Convert logits to characters.
    greedy_prediction = logits.argmax(2).detach().cpu().numpy().T
    greedy_prediction = greedy_prediction.squeeze(-1)

    all_stroke_x_points = [
        stroke_data.x_points for stroke_data in all_stroke_data
    ]
    all_stroke_y_points = [
        stroke_data.y_points for stroke_data in all_stroke_data
    ]

    # Return the predicted string and the x and y points of the strokes and bezier curves.
    return (
        train.greedy_decode(greedy_prediction),
        all_stroke_x_points,
        all_stroke_y_points,
        all_bezier_x_points,
        all_bezier_y_points,
    )
