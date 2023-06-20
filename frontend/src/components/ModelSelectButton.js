import React from "react";

const ModelSelectButton = ({ modelName, selectedModel, setSelectedModel }) => {
    const modelButtonStyle = {
        border: "2px solid black",
        boxShadow: "none",
        margin: "0 60px",
        padding: "20px",
        borderRadius: "5px",
        fontSize: "200%",
        fontWeight: 'bolder',
      };
    
      const selectedModelButtonStyle = {
        ...modelButtonStyle,
        backgroundColor: "green",
      };

      
    return (
      <button
        style={
          modelName === selectedModel
            ? selectedModelButtonStyle
            : modelButtonStyle
        }
        onClick={() => setSelectedModel(modelName)}
      >
        {modelName}
      </button>
    );
  };

  export default ModelSelectButton;