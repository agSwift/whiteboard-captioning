import React from "react";

const PredictButton = ({ onClick }) => {
  const style = {
    border: "2px solid black",
    boxShadow: "none",
    padding: "20px",
    borderRadius: "5px",
    fontSize: "200%",
    fontWeight: "bolder",
    backgroundColor: "lightyellow",
  };
  
  return (
    <button onClick={onClick} style={style}>
      PRESS TO PREDICT
    </button>
  );
};

export default PredictButton;
