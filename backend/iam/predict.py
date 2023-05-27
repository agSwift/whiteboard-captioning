from functools import lru_cache
from pathlib import Path
import torch

from iam import bezier_curves, train

ALL_NUM_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ALL_BEZIER_CURVE_DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ALL_MODEL_NAMES = [model_type.name for model_type in train.ModelType]


@lru_cache(maxsize=1)
def _load_models():
    loaded_models = {}

    for model_name in ALL_MODEL_NAMES:
        for bezier_curve_degree in ALL_BEZIER_CURVE_DEGREES:
            for num_layers in ALL_NUM_LAYERS:
                for bidirectional in [True]:
                    # Create the model file name.
                    model_file_name = train.get_model_file_name(
                        model_name=model_name.lower(),
                        bezier_curve_degree=bezier_curve_degree,
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                    )

                    # Ensure the model exists.
                    model_path = Path(f"iam/models/{model_file_name}.ckpt")
                    assert model_path.exists(), (
                        f"Model {model_file_name} does not exist.\n"
                        "Run train.py to train the model."
                    )

                    # Load the model.
                    loaded_models[model_file_name] = torch.load(model_path)

    return loaded_models


LOADED_MODELS = _load_models()


def _parse_stroke(stroke: list[dict[str, float]]) -> bezier_curves.StrokeData:
    """Parse a stroke into a bezier_curves.StrokeData object.

    Args:
        stroke (list[dict[str, float]]): A stroke, which is a list of points.
            Each point is a dictionary with keys "x", "y", and "time".

    Returns:
        bezier_curves.StrokeData: A bezier_curves.StrokeData object.

    Raises:
        AssertionError: If the number of x points, y points, time stamps, and pen ups are not equal.
    """
    x_points = []
    y_points = []
    time_stamps = []
    pen_ups = []
    fst_timestamp = 0

    for i, point in enumerate(stroke):
        x_points.append(point["x"])
        y_points.append(point["y"])

        if not time_stamps:
            # Ensure first time stamp is 0, not the actual time stamp.
            fst_timestamp = point["time"]
            time_stamps.append(0)
        else:
            time_stamps.append(point["time"] - fst_timestamp)

        if i == len(stroke) - 1:
            # If this is the last point in the stroke, the pen is up.
            pen_ups.append(1)
        else:
            # If this is not the last point in the stroke, the pen is down.
            pen_ups.append(0)

    assert (
        len(x_points) == len(y_points) == len(time_stamps) == len(pen_ups)
    ), (
        f"Invalid stroke: {stroke}\n"
        "The number of x points, y points, time stamps, and pen ups must be equal."
    )

    return bezier_curves.StrokeData(
        x_points=x_points,
        y_points=y_points,
        time_stamps=time_stamps,
        pen_ups=pen_ups,
    )


def greedy_predict(
    strokes: list[list[dict[str, float]]],
    max_y: float,
    model_name: str,
    bezier_curve_degree: int,
    num_layers: int,
    bidirectional: bool,
) -> str:
    """Predict the characters in the strokes.

    Uses greedy decoding to convert the logits to a string.

    Args:
        strokes (list[list[dict[str, float]]]): A list of strokes. Each stroke is a list of points.
            Each point is a dictionary with keys "x", "y", and "time".
        max_y (float): The maximum y value that can be reached in the writing area.
        model_name (str): The name of the model to use.
        bezier_curve_degree (int): The degree of the bezier curves to fit to the strokes.
        num_layers (int): The number of recurrent layers in the model.
        bidirectional (bool): Whether the model is bidirectional.

    Returns:
        str: The predicted string.

    Raises:
        ValueError: If the model name is invalid.
        AssertionError: If the number of x points, y points, time stamps, and pen ups are not equal.
        AssertionError: If any y point is not between 0 and 1.
        AssertionError: If the model does not exist.
    """
    if model_name not in ALL_MODEL_NAMES:
        raise ValueError(
            f"Invalid model name: {model_name}\n"
            f"Must be one of: {ALL_MODEL_NAMES}"
        )

    # Parse the strokes into bezier_curves.StrokeData objects.
    all_stroke_data = []

    for stroke in strokes:
        stroke_data = _parse_stroke(stroke)

        # Normalise the stroke data.
        min_x = min(stroke_data.x_points)
        min_y = min(stroke_data.y_points)

        # Shift x and y points to start at 0.
        stroke_data.x_points = [x - min_x for x in stroke_data.x_points]
        stroke_data.y_points = [y - min_y for y in stroke_data.y_points]

        # Scale points so that the y_points are between 0 and 1. The x_points will be scaled
        # by the same amount, to preserve the aspect ratio.
        scale = (
            1 if max_y - min_y == 0 else 1 / (max_y - min_y)
        )  # Avoid division by 0.

        stroke_data.x_points = [x * scale for x in stroke_data.x_points]
        stroke_data.y_points = [y * scale for y in stroke_data.y_points]

        assert all(0 <= y <= 1 for y in stroke_data.y_points), (
            f"Invalid stroke: {stroke}\n"
            f"All y-points must be between 0 and 1."
        )

        all_stroke_data.append(stroke_data)

    # Fit bezier curves to the strokes.
    bezier_curves_data = [
        bezier_curves.fit_stroke_with_bezier_curve(
            stroke=stroke_data, degree=bezier_curve_degree
        )
        for stroke_data in all_stroke_data
    ]

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
    return train.greedy_decode(greedy_prediction)
