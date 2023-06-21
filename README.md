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

![writing](https://github.com/agSwift/whiteboard-captioning/assets/36814369/ae8c2fe3-7b58-4a43-91d9-04f88fd38665)
![bezier](https://github.com/agSwift/whiteboard-captioning/assets/36814369/18464e6e-0574-459f-84a4-f9168773d79a)

## License

This project uses the following license: [MIT](https://choosealicense.com/licenses/mit/)
