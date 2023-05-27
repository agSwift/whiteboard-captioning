from flask import Flask, request, jsonify
from flask_cors import CORS

from iam import predict

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["POST"])
def post_data():
    assert request.is_json, "Request must be JSON."
    data = request.get_json()

    assert "strokes" in data, "Request must contain strokes."
    assert "max_y" in data, "Request must contain max_y."
    assert "model_name" in data, "Request must contain model_name."
    assert (
        "bezier_curve_degree" in data
    ), "Request must contain bezier_curve_degree."
    assert "num_layers" in data, "Request must contain num_layers."

    strokes = data["strokes"]
    max_y = data["max_y"]
    model_name = data["model_name"]
    bezier_curve_degree = data["bezier_curve_degree"]
    num_layers = data["num_layers"]
    bidirectional = data["bidirectional"]

    assert isinstance(strokes, list), "Strokes must be a list."
    assert len(strokes) > 0, (
        "Strokes must contain at least one stroke. Each stroke must be a list of points."
        "Each point must be a dictionary with keys 'x', 'y', and 'time'."
    )
    assert isinstance(max_y, (int, float)), "max_y must be a number."
    assert isinstance(model_name, str), "model_name must be a string."
    assert isinstance(
        bezier_curve_degree, int
    ), "bezier_curve_degree must be an integer."
    assert isinstance(num_layers, int), "num_layers must be an integer."
    assert isinstance(bidirectional, bool), "bidirectional must be a boolean."

    prediction = predict.greedy_predict(
        strokes=strokes,
        max_y=max_y,
        model_name=model_name,
        bezier_curve_degree=bezier_curve_degree,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    return jsonify(prediction=prediction)
