## Self-Driving Perception Demo (YOLO-Based)

This project is a hackathon-ready demo for a self-driving car perception system built around a YOLO object detection model. It includes:

- A **training pipeline** to fine-tune YOLO on your own driving dataset with **class-balanced sampling**.
- A **web UI** to upload any road image (not just from the dataset) and visualize detections.
- Several **unique features** aimed at explainability and safety analysis.

### Key Features

- **YOLO-based detector** for cars, pedestrians, traffic lights, signs, etc.
- **Balanced training**: optional class-reweighting / sampling to counter class imbalance.
- **Works on arbitrary images**: upload any street/traffic image; the model runs inference and shows outputs.
- **Risk heatmap**: heuristic risk score per region (e.g. close pedestrians, oncoming vehicles, red lights).
- **Scenario summary**: short natural-language description of the scene (e.g. “Urban intersection, pedestrian crossing from right, green light”).
- **Dataset lens (optional)**: analyze your dataset’s class distribution and typical failure cases.

### Tech Stack

- **Backend / ML**: Python, [Ultralytics YOLO](https://docs.ultralytics.com/), PyTorch.
- **Frontend UI**: Python web UI (Gradio) for fast, professional-looking interfaces, with custom styling.

If you prefer a separate React/TypeScript frontend, you can add one later and call the Python backend via REST; the core model and API are designed with that in mind.

### Project Layout

- `training/`
  - `train_yolo.py` – training + validation entrypoint.
  - `data_utils.py` – class distribution, sampling weights, basic diagnostics.
- `app/`
  - `inference_service.py` – model loading and inference utilities.
  - `ui.py` – web UI for uploads, visualization, and unique features.
- `models/` – trained YOLO weights (`best.pt`, etc.).
- `data/` – your dataset and YOLO data config.
- `requirements.txt` – Python dependencies.

### Quick Start

1. **Create a virtual environment and install deps**

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

2. **Prepare a YOLO-format dataset**

- Place images and labels under `data/`.
- Create a `data/config.yaml` describing train/val paths and class names (YOLOv8-style).

3. **Train the model**

```bash
python training/train_yolo.py --data data/config.yaml --epochs 50 --imgsz 640 --balance
```

4. **Run the web UI**

```bash
python app/ui.py
```

This will open a browser interface where you can upload any road image and see:

- YOLO detections.
- Risk score & heatmap overlay.
- Scenario summary and basic explanation of what the model is focusing on.

# selfdriving
