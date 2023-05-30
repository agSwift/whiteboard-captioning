from flask import Flask, request, jsonify
from flask_cors import CORS

from iam import predict

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["POST"])
def post_data():
    data = request.get_json()

    assert "strokes" in data, "Request must contain strokes."
    assert "model_name" in data, "Request must contain model_name."
    assert (
        "bezier_curve_degree" in data
    ), "Request must contain bezier_curve_degree."
    assert "num_layers" in data, "Request must contain num_layers."

    strokes = data["strokes"]
    model_name = data["model_name"]
    bezier_curve_degree = data["bezier_curve_degree"]
    num_layers = data["num_layers"]
    bidirectional = data["bidirectional"]

    assert isinstance(strokes, list), "Strokes must be a list."
    assert isinstance(model_name, str), "model_name must be a string."
    assert isinstance(
        bezier_curve_degree, int
    ), "bezier_curve_degree must be an integer."
    assert isinstance(num_layers, int), "num_layers must be an integer."
    assert isinstance(bidirectional, bool), "bidirectional must be a boolean."

    prediction = predict.greedy_predict(
        strokes=strokes,
        model_name=model_name,
        bezier_curve_degree=bezier_curve_degree,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    return jsonify(prediction=prediction)
