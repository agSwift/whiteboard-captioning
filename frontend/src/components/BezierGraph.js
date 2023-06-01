import React from "react";

import {
  Line,
  XAxis,
  YAxis,
  Tooltip,
  LineChart,
  CartesianGrid,
  ComposedChart,
  Scatter,
} from "recharts";

const BezierGraph = ({
  strokeXPoints,
  strokeYPoints,
  bezierXPoints,
  bezierYPoints,
  graphNumber,
}) => {
  const strokePoints = [];
  const bezierPoints = [];
  if (graphNumber >= 0) {
    strokeXPoints[graphNumber].forEach((val, index) =>
      strokePoints.push({
        x: strokeXPoints[graphNumber][index],
        y: strokeYPoints[graphNumber][index],
      })
    );

    bezierXPoints[graphNumber].forEach((val, index) =>
      bezierPoints.push({
        x: bezierXPoints[graphNumber][index],
        y: bezierYPoints[graphNumber][index],
      })
    );
  }

  return (
    <ComposedChart width={400} height={400} b>
      <CartesianGrid strokeDasharray={"3 3"}/>
      <Line xAxisId="bezier" yAxisId="bezier" dataKey="y" dot={false}  data={bezierPoints} />
      <Scatter xAxisId="stroke" yAxisId="stroke" dataKey="y" dot={false} fill='blue'  data={strokePoints} />
  </ComposedChart>
  );
};

export default BezierGraph;
