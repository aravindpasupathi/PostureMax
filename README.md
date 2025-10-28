## Posture AI – Webcam Posture Detection

A lightweight posture tracking app built during the AI Research Hackathon at UNC Charlotte. Uses MediaPipe Pose and a simple classifier to detect good vs bad posture from a webcam stream. Includes tools to collect your own dataset and train a model.

### Features
- Real‑time webcam posture prediction
- Data collection script to label frames as good/bad
- Simple training script (Decision Tree) with saved model

### Repository Structure
- `posture/` – Python package with reusable modules
  - `geometry.py` – simple `Point` and angle utils
  - `collect.py` – `run_data_collector()` to capture labeled frames
  - `predict.py` – `run_live_predictor()` for real‑time inference
  - `train.py` – `train_model()` to train and save a model
- `data/` – dataset location (CSV), kept out of Git
- `models/` – trained model artifacts, kept out of Git
- `live_posture_predictor.py` – Thin wrapper that runs the live predictor
- `Python_data_collector.py` – Thin wrapper that runs the collector
- `train_model_v3.py` – Thin wrapper that trains the model

### Quickstart
1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# On Windows: .venv\Scripts\activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Collect labeled data (optional, if you want to retrain)
```bash
python Python_data_collector.py
```
- Press `g` for good posture, `b` for bad posture, `q` to quit
- Data will append to `data/posture_dataset.csv`

4) Train the model (optional)
```bash
python train_model_v3.py
```
- Outputs `models/posture_model_v3.pkl`

5) Run the live predictor
```bash
python live_posture_predictor.py
```
- Press `q` to quit

### Model Notes
- The included trainer uses a Decision Tree for simplicity. You can swap in any sklearn model.
- Features are the raw 3D pose landmark coordinates from MediaPipe; you can experiment with engineered angles.

### Requirements
- Python 3.9+
- Webcam access

### Troubleshooting
- If the webcam doesn’t open, try changing the camera index in `cv2.VideoCapture(0)` to `1`.
- If MediaPipe install fails on Apple Silicon, ensure `pip` is up to date and Python is from python.org or use `conda`.

### License
This project is released under the MIT License. See `LICENSE`.
