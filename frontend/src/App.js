import React, { useState } from "react";
import DrawingCanvas from "./components/DrawingCanvas";
import ModelSelectButton from "./components/ModelSelectButton";
import PredictButton from "./components/PredictButton";
import ParameterSlider from "./components/ParameterSlider";
import ToggleSwitch from "./components/ToggleSwitch";

const BACKEND_URL = "http://127.0.0.1:5000/";
const PREDICTION_MODELS = ["RNN", "LSTM", "GRU"];

function App() {
  const [predictedText, setPredictedText] = useState("Predicted text here");

  const [strokes, setStrokes] = useState([]);
  const [maxY, setMaxY] = useState(0);
  const [maxX, setMaxX] = useState(0);

  const [selectedModel, setSelectedModel] = useState(PREDICTION_MODELS[1]);
  const [bezierCurveDegree, setBezierCurveDegree] = useState(5);
  const [numLayers, setNumLayers] = useState(3);
  const [bidirectional, setBidirectional] = useState(true);

  const SendData = (e) => {
    const data = {
      strokes: strokes,
      max_y: maxY,
      model_name: selectedModel,
      bezier_curve_degree: parseInt(bezierCurveDegree),
      num_layers: parseInt(numLayers),
      bidirectional: bidirectional,
    };

    console.log(data);
    fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then(async (response) => {
        const body = await response.json();
        console.log(body);
        setPredictedText(body.prediction);
      })
      .catch(console.log);
  };

  const centeredDivStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    marginTop: "20px",
  };

  const borderedDivStyle = {
    ...centeredDivStyle,
    border: "2px solid black",
    width: "50%",
    marginLeft: "25%",
    borderRadius: "10px",
    fontSize: "200%",
  };

  return (
    <div>
      <div style={centeredDivStyle}>
        <DrawingCanvas
          strokes={strokes}
          setStrokes={setStrokes}
          maxX={maxX}
          setMaxX={setMaxX}
          maxY={maxY}
          setMaxY={setMaxY}
        />
      </div>
      <div
        style={borderedDivStyle}
      >
        {predictedText}
      </div>
      <div style={centeredDivStyle}>
        <PredictButton onClick={SendData} />
      </div>
      <div
        style={centeredDivStyle}
      >
        {PREDICTION_MODELS.map((model) => (
          <ModelSelectButton
            key={model}
            modelName={model}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
          />
        ))}
      </div>
      <div style={centeredDivStyle}>
        <ParameterSlider
          min={1}
          max={9}
          value={bezierCurveDegree}
          setValue={setBezierCurveDegree}
          label={"Bezier Curve Degree"}
        />
        <ParameterSlider
          min={1}
          max={9}
          value={numLayers}
          setValue={setNumLayers}
          label={"Number of Layers"}
        />
        <ToggleSwitch label={'Bidirectional?'} setValue={setBidirectional} value={bidirectional} />
      </div>
    </div>
  );
}

export default App;
