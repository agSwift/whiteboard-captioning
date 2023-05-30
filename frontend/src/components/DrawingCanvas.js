import React, { useState, useRef, useEffect } from "react";

const DrawingCanvas = ({
  strokes,
  setStrokes,
  maxX,
  setMaxX,
  maxY,
  setMaxY,
}) => {
  const canvasRef = useRef(null);
  const [canvasContext, setCanvasContext] = useState(null);

  const [color, setColor] = useState("#000000");
  const [size, setSize] = useState(10);
  const [canvasMousePosition, setCanvasMousePosition] = useState({
    x: 0,
    y: 0,
  });
  const [isNewStroke, setIsNewStroke] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const maxY = window.innerHeight / 2;
    const maxX = window.innerWidth * 0.75;

    canvas.height = maxY;
    canvas.width = maxX;

    setMaxY(maxY);
    setMaxX(maxX);
    setCanvasContext(ctx);
  }, [canvasRef]);

  const getCanvasX = (x) => x - canvasRef.current.offsetLeft;

  const getCanvasY = (y) => y - canvasRef.current.offsetTop;

  const Draw = (e) => {
    const newPoints = strokes;

    // If left button is clicked
    if (e.buttons !== 1) {
      if (!isNewStroke) {
        setIsNewStroke(true);
      }
      return;
    } else if (isNewStroke) {
      // Starting a new stroke means we should add a new line segment in points
      newPoints.push([
        {
          x: canvasMousePosition.x,
          y: canvasMousePosition.y,
          time: Date.now() / 1000,
        },
      ]);
      setIsNewStroke(false);
    }

    const ctx = canvasContext;
    const canvasEventPosition = {
      x: getCanvasX(e.clientX),
      y: getCanvasY(e.clientY),
    };

    ctx.beginPath();
    // Draw the line
    ctx.moveTo(canvasMousePosition.x, canvasMousePosition.y);
    ctx.lineTo(canvasEventPosition.x, canvasEventPosition.y);

    // Update canvas mouse position
    setCanvasMousePosition(canvasEventPosition);

    // Stroke characteristics
    ctx.strokeStyle = color;
    ctx.lineWidth = size;
    ctx.lineCap = "round";
    ctx.stroke();

    // Add moved to point
    const coordPointX = canvasEventPosition.x;
    const coordPointY = maxY - canvasEventPosition.y; //html Y is top down, we need Y as bottom to top
    if (
      coordPointX >= 0 &&
      coordPointX <= maxX &&
      coordPointY >= 0 &&
      coordPointY <= maxY
    ) {
      newPoints[newPoints.length - 1].push({
        x: coordPointX,
        y: coordPointY,
        time: Date.now() / 1000,
      });
    }

    // Update all the points
    setStrokes(newPoints);
  };

  const SetPos = (e) => {
    setCanvasMousePosition({
      x: getCanvasX(e.clientX),
      y: getCanvasY(e.clientY),
    });
  };

  const style = {
    border: "2px solid black",
    borderRadius: "10px",
  };

  return (
    <canvas
      ref={canvasRef}
      onMouseEnter={(e) => SetPos(e)}
      onMouseMove={(e) => {
        SetPos(e);
        Draw(e);
      }}
      onMouseDown={(e) => SetPos(e)}
      style={style}
    />
  );
};

export default DrawingCanvas;
