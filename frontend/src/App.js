import React, { useEffect, useRef, useState } from "react"; 

const BACKEND_URL = 'http://127.0.0.1:5000/';
const PREDICTION_MODELS = ['RNN', 'LSTM', 'GRU'];

function App() {
    const [canvasMousePosition, setCanvasMousePosition] = useState({ x: 0, y: 0 });
    const [isNewStroke, setIsNewStroke] = useState(true);
    const canvasRef = useRef(null);
    const [canvasContext, setCanvasContext] = useState(null);
    const [color, setColor] = useState("#000000");
    const [size, setSize] = useState(10);
    const [predictedText, setPredictedText] = useState('Predicted text here');

    const [strokes, setStrokes] = useState([]);
    const [maxY, setMaxY] = useState(0);
    const [maxX, setMaxX] = useState(0);
    const [selectedModel, setSelectedModel] = useState(PREDICTION_MODELS[1]);
    const [bezierCurveDegree, setBezierCurveDegree] = useState(5);
    const [numLayers, setNumLayers] = useState(3);
    const [bidirectional, setBidirectional] = useState(true);

    
    
    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const maxY = window.innerHeight / 2
        const maxX = window.innerWidth * 0.75;

        canvas.height = maxY;
        canvas.width = maxX;

        setMaxY(maxY);
        setMaxX(maxX);
        setCanvasContext(ctx);
    }, [canvasRef]);

    const SetPos = (e) => {
        setCanvasMousePosition({
            x: getCanvasX(e.clientX),
            y: getCanvasY(e.clientY),
        });
    };

    const SendData = (e) => {
        const data = {
            'strokes': strokes,
            'max_y': maxY,
            'model_name': selectedModel,
            'bezier_curve_degree': parseInt(bezierCurveDegree),
            'num_layers': parseInt(numLayers),
            'bidirectional': bidirectional,
        }

        console.log(data)
        fetch(BACKEND_URL, {
            method: "POST",
            headers: {
              Accept: "application/json",
              "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        }).then(async (response)=>{
            const body = await response.json();
            console.log(body);
            setPredictedText(body.prediction);
        }).catch(console.log);
    }

    const getCanvasX = (x) => x - canvasRef.current.offsetLeft;

    const getCanvasY = (y) => y - canvasRef.current.offsetTop;

    const Draw = (e) => {
        const newPoints = strokes;

        // If left button is clicked
        if (e.buttons !== 1) {
            if(!isNewStroke) {
                setIsNewStroke(true);
            }
            return;
        } else if (isNewStroke){
            // Starting a new stroke means we should add a new line segment in points
            newPoints.push([{x: canvasMousePosition.x, y:  canvasMousePosition.y, time: Date.now() / 1000}])
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
        ctx.lineTo(canvasEventPosition.x,  canvasEventPosition.y);

        // Update canvas mouse position
        setCanvasMousePosition(canvasEventPosition);

        // Stroke characteristics
        ctx.strokeStyle = color;
        ctx.lineWidth = size;
        ctx.lineCap = "round";
        ctx.stroke();

        // Add moved to point
        const coordPointX =  canvasEventPosition.x
        const coordPointY =  maxY - canvasEventPosition.y //html Y is top down, we need Y as bottom to top
        if(coordPointX >= 0 && coordPointX <= maxX && coordPointY >=0 && coordPointY <= maxY) {
            newPoints[newPoints.length-1].push({x: coordPointX, y: coordPointY, time: Date.now() / 1000})
        }

        // Update all the points
        setStrokes(newPoints)
    };

    const modelButtonStyle = {
        border: '2px solid black',
        boxShadow: 'none',
        margin: '0 10px',
        padding: '10px',
        borderRadius: '5px',
        fontSize: '15px',
    };

    const selectedModelButtonStyle = {
        border: '2px solid black',
        boxShadow: 'none',
        margin: '0 10px',
        padding: '10px',
        borderRadius: '5px',
        fontSize: '15px',
        backgroundColor: 'darkgreen',
    };

    const divStyle = {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        marginTop: '50px',
    };

    const ModelSelectButton = ({modelName}) => {
        return <button style={modelName === selectedModel ? selectedModelButtonStyle:modelButtonStyle} onClick={()=>setSelectedModel(modelName)}>
                {modelName}
            </button>
        };

    return (
        <div>
            <div
                style={divStyle}
            >
                <canvas
                    ref={canvasRef}
                    onMouseEnter={(e) => SetPos(e)}
                    onMouseMove={(e) => {SetPos(e); Draw(e)}}
                    onMouseDown={(e) => SetPos(e)}
                    style={{
                        border: '2px solid black',
                        borderRadius: '10px',
                    }}
                ></canvas>
            </div>
            <div
                style={{
                ...divStyle,
                border: '2px solid black',
                width: '50%',
                marginLeft: '25%',
                borderRadius: '10px',
                fontSize: '200%',
                }}
            >
                {predictedText}
            </div>
            <div style={divStyle}>
                <button onClick={SendData} style={{backgroundColor: 'lightyellow', fontWeight: 'bolder'}}>PRESS TO PREDICT</button>
            </div>
            <div
                style={{
                    ...divStyle,        
                }}
            >
                {PREDICTION_MODELS.map((model)=><ModelSelectButton modelName={model} />)}
            </div>
            <div style={divStyle}>
                <div style={{width: "100%", padding: '0 20px'}}>
                    <input
                        type="range"
                        step={1}
                        valueLabelDisplay="on"
                        max={9}
                        value={bezierCurveDegree}
                        onChange={(e)=>setBezierCurveDegree(e.target.value)}
                        />
                    <p>Bezier Curve Degree: {String(bezierCurveDegree)} </p>
                </div>
                <div style={{width: "100%", padding: '0 20px'}}>
                <input
                        type="range"
                        step={1}
                        valueLabelDisplay="on"
                        max={9}
                        value={numLayers}
                        onChange={(e)=>setNumLayers(e.target.value)}
                        />
                    <p>Number of Layers: {String(numLayers)} </p>
                </div>
                <div style={{width: "100%", padding: '0 20px', alignItems: "center", justifyContent: "center"}}>
                <input
                        type="range"
                        step={0}
                        valueLabelDisplay="on"
                        max={1}
                        value={bidirectional ? 1: 0}
                        onChange={(e)=>setBidirectional(e.target.value==1)}
                        />
                    <p>Bidirectional: {String(bidirectional)} </p>
                </div>
            </div>
        </div>
    );
}

export default App;