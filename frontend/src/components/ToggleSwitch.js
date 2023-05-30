import React from "react";
import "./ToggleSwitch.css";
import "./ParameterSlider.css";

const ToggleSwitch = ({ value, setValue, label }) => {
  const containerStyle = { width: "100%", padding: "0 20px" };
  const labelStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "bolder",
  };

  return (
    <div style={containerStyle}>
      <p style={labelStyle}>{label}</p>
      <div style={labelStyle}>
      <label class="switch" style={{containerStyle}}>
        <input
          type="checkbox"
          checked={value}
          onChange={(_) => setValue(!value)}
        />
        <span class="slider round"></span>
      </label>
      </div>
      <span id="rangeValue">{String(value)}</span>
    </div>
  );
};

export default ToggleSwitch;
