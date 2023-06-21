# Whiteboard Captioning

This project was developed as part of my final year MEng Computing degree project at Imperial College London ([thesis](https://github.com/agSwift/whiteboard-captioning/blob/04b792d451adea5e1373f36f3de8461ab61d3869/final_report.pdf)). It explores an approach to handwriting recognition that leverages dynamic data from stylus or fingertip movements, and introduces two Bézier curve feature sets for the transformation of stroke points.


## Prerequisites

Before you begin, ensure you have installed the following:
* A version of [`Python 3.9.7`](https://www.python.org/downloads/release/python-397/) and above.
* A version of [`Node v14.7.0`](https://nodejs.org/en/blog/release/v14.7.0) and above.

## Getting Started
Download the [IAM-OnDB dataset](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database). Save the `ascii`, `lineImages` and `lineStrokes` directories under the `backend/datasets/IAM` folder within the repository.

Once done, ensure that you have navigated (`cd`) into the root of the repository.

To install all dependencies, run:
```bash
  pip install -r requirements.txt
  cd frontend && npm install
  cd ..  # Make sure to return to the root directory after installing frontend dependencies.
```

To ensure all start-up scripts are executable, run:
```bash
  chmod +x /backend/extract.sh   
  chmod +x /backend/start.sh 
  chmod +x /frontend/start.sh   
```

To extract and preprocess all data required for the application, run:
```bash
  ./backend/extract.sh 
```

To start the backend, run:
```bash
  ./backend/start.sh 
```

To start the frontend, open a new terminal session and run:
```bash
  ./frontend/start.sh 
```

Open your favourite browser, and go to http://localhost:3000.


## Demo

![writing_AdobeExpress](https://github.com/agSwift/whiteboard-captioning/assets/36814369/c699c269-6741-4863-acb7-65cf509bf45b)

![bezier_AdobeExpress](https://github.com/agSwift/whiteboard-captioning/assets/36814369/d5562d86-b8fb-4dc9-b8f1-478ab70ed4a4)

## License

This project uses the following license: [MIT](https://choosealicense.com/licenses/mit/)
