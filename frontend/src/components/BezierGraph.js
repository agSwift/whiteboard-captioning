import React from "react";

import {
  Line,
  XAxis,
  YAxis,
  Tooltip,
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
      <Line xAxisId="strokes" yAxisId="strokes" dataKey="y" data={bezierPoints} dot={false} />
      <Scatter xAxisId="strokes" yAxisId="strokes" dataKey="y" fill={'red'} data={strokePoints} />
      <XAxis xAxisId="strokes" orientation="bottom"  type="number" dataKey="x" tick={true} />
      <YAxis yAxisId="strokes" orientation='left' type="number" tick={true} reversed={true} />
      <Tooltip />
  </ComposedChart>
  );
};

export default BezierGraph;