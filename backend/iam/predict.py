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
                    model.load_state_dict(torch.load(model_path))

                    # Add the model to the dictionary.
                    loaded_models[model_file_name] = model

    return loaded_models


LOADED_MODELS = _load_models()


def _parse_stroke(*, stroke: list[dict[str, float]], bezier_curve_degree: int, points_per_second: int = 20) -> bezier_curves.StrokeData:
    """Parse a stroke into a bezier_curves.StrokeData object.

    Args:
        stroke (list[dict[str, float]]): A stroke, which is a list of points.
            Each point is a dictionary with keys "x", "y", and "time".
        bezier_curve_degree (int): The degree of the bezier curves to fit to the strokes.
        points_per_second (int): The number of points to keep per second.

    Returns:
        bezier_curves.StrokeData: A bezier_curves.StrokeData object.

    Raises:
        AssertionError: If the number of x points, y points, time stamps, and pen ups are not equal.
    """
    x_points = []
    y_points = []
    time_stamps = []
    pen_ups = []

    total_time = stroke[-1]['time'] - stroke[0]['time']
    total_points = int(total_time * points_per_second)

    # Ensure total points is a multiple of bezier_curve_degree + 1
    total_points = total_points - (total_points % (bezier_curve_degree + 1))
    indices_to_keep = np.linspace(0, len(stroke)-1, total_points, dtype=int)

    for i in indices_to_keep:
        point = stroke[i]
        x_points.append(point["x"])
        y_points.append(point["y"])
        time_stamps.append(point["time"])

        if i == indices_to_keep[-1]:
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
    if model_name not in [model_type.name for model_type in train.ModelType]:
        raise ValueError(
            f"Invalid model name: {model_name}\n"
            f"Valid model names: {[model_type.name for model_type in train.ModelType]}"
        )

    # Parse the strokes into bezier_curves.StrokeData objects.
    all_stroke_data = []

    for stroke in strokes:
        stroke_data = _parse_stroke(stroke=stroke, bezier_curve_degree=bezier_curve_degree)

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
    bezier_curves_data = np.array(
        [
            bezier_curves.fit_stroke_with_bezier_curve(
                stroke=stroke_data, degree=bezier_curve_degree
            )
            for stroke_data in all_stroke_data
        ]
    )

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

    return train.greedy_decode(greedy_prediction)
