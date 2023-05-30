import React from "react";
import "./ParameterSlider.css";

const ParameterSlider = ({ min, max, value, setValue, label }) => {
  const containerStyle = { width: "100%", padding: "0 20px" };
  const sliderStyle = { width: "100%" };
  const labelStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "bolder",
  };

  return (
    <div style={containerStyle}>
      <p style={labelStyle}>{label}</p>
      <input
        type="range"
        step={1}
        valueLabelDisplay="on"
        min={min}
        max={max}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        style={sliderStyle}
      />
      <span id="rangeValue">{String(value)}</span>
    </div>
  );
};

export default ParameterSlider;
