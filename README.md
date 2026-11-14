# Real-Time Facial Emotion Recognition
### Deep Learning System Powered by ANN & CNN with FastAPI Backend

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

---

## 📋 What This Project Is All About

This project is an **end-to-end Affective Computing & Deep Learning system** designed to detect and classify human facial expressions in real time from static images, uploaded videos, and live webcam feeds.

The system detects **7 canonical human emotions**:

| Emotion | Label | Visual Features & Morphological Markers |
|:---|:---:|:---|
| 😠 **Angry** | `angry` | Knitted eyebrows, lowered brow, intense stare, tightened lips |
| 🤢 **Disgust** | `disgust` | Wrinkled nose bridge, raised upper lip, narrowed eyes |
| 😨 **Fear** | `fear` | Widened eye aperture, raised inner eyebrows, slightly parted mouth |
| 😊 **Happy** | `happy` | Elevated cheek zygomatic muscles, broad smile, crinkled eye corners |
| 😐 **Neutral** | `neutral` | Relaxed facial musculature, horizontal lip line, resting gaze |
| 😢 **Sad** | `sad` | Downturned mouth corners, inner eyebrows pulled upwards |
| 😲 **Surprise** | `surprise` | Vertically dropped jaw, wide circular eye aperture, arched brows |

### 🔬 Core Scientific & Engineering Highlights

1. **Dual-Model Architectural Comparison:**
   - **Baseline ANN (Multi-Layer Perceptron):** A dense neural network compared with the CNN while 2D spatial locality is discarded.
   - **Primary CNN:** Uses a compact four-block convolutional network with training-only image augmentation and dropout. Test metrics are measured from actual predictions and depend on the data and training run.
2. **Real-Time Computer Vision Pipeline:**
   - Uses OpenCV's **Haar Feature-Based Cascade Classifier** (`haarcascade_frontalface_default.xml`) to detect and isolate facial regions in $\mathcal{O}(1)$ time via integral images.
   - Deterministic preprocessing converts faces to single-channel Grayscale at $64 \times 64$ resolution with $[0.0, 1.0]$ normalization.
   - Inference latency is measured during evaluation and depends on the hardware.
3. **Softmax Probabilities:**
   - Returns the model's softmax probabilities for all 7 classes.
4. **Production-Ready Full-Stack Serving:**
   - High-throughput asynchronous **FastAPI** backend with CORS middleware and auto-generated OpenAPI documentation.
   - Interactive, responsive **Dark Glassmorphism UI** with three dedicated viewports:
     - **Home / Prediction Studio:** Live image upload, webcam detection, model toggling (ANN vs. CNN), confidence scores, and probability meters.
     - **Visual Analytics:** Dataset class distribution, training curves, confusion matrices, and sample predictions.
     - **Model Evaluation:** Side-by-side performance comparison tables and detailed classification reports.
5. **Comprehensive Technical Report:**
   - Contains a complete 24-page publication-quality engineering report compiled to **`report.pdf`** and **`report.html`**.

---

## ⚡ Execution Commands & Running the Project

Follow these step-by-step instructions to run the application on your machine.

### Step 1: Open Your Terminal
Open PowerShell or Command Prompt (CMD) in the project root directory:
```bash
cd "c:\Users\Anurag Rajendra\OneDrive\Pictures\PROJECTS\REAL TIME FACIAL EMOTION RECOGNITION"
```

---

### Step 2: Activate the Virtual Environment

Your virtual environment is already installed in the `venv/` directory. Activate it based on your shell:

#### 🔹 Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
> **Tip for PowerShell Users:** If you see an error saying *`Execution of scripts is disabled on this system`*, run this command once in your PowerShell window:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

#### 🔹 Windows Command Prompt (CMD):
```cmd
venv\Scripts\activate.bat
```

#### 🔹 macOS / Linux (if applicable):
```bash
source venv/bin/activate
```

*(Once activated, you will see `(venv)` prefixed in your terminal prompt).*

---

### Step 3: Run the Web Server (Uvicorn / Localhost)

You have two simple ways to launch the server:

#### Option A: Direct Uvicorn Command (Recommended for Development)
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
- `--reload`: Automatically restarts the server whenever code changes are saved.
- `--host 127.0.0.1`: Binds the server to localhost.
- `--port 8000`: Runs on port 8000.

#### Option B: Run via Python Script
```bash
python app.py
```
*(This triggers the `uvicorn.run(...)` entrypoint defined at the bottom of `app.py`).*

#### Option C: One-Line Direct Execution (Without Manual Activation)
If you do not want to activate the venv first, run directly via the venv executable:
```powershell
.\venv\Scripts\uvicorn.exe app:app --reload --host 127.0.0.1 --port 8000
```

---

### Step 4: Access Localhost in Your Browser

Once the terminal outputs `Application startup complete` and `Uvicorn running on http://127.0.0.1:8000`:

| Interface | URL | Description |
|:---|:---|:---|
| 🏠 **Web Application (UI)** | **[http://localhost:8000](http://localhost:8000)** | Prediction Studio, Analytics & Evaluation dashboards |
| 📊 **Analytics Dashboard** | **[http://localhost:8000/analysis](http://localhost:8000/analysis)** | Training curves, dataset charts & confusion matrices |
| ⚖️ **Model Evaluation** | **[http://localhost:8000/evaluation](http://localhost:8000/evaluation)** | Measured test metrics & comparisons |
| 📖 **Interactive Swagger Docs** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | Test the REST API endpoints directly in your browser |
| 📑 **ReDoc Documentation** | **[http://localhost:8000/redoc](http://localhost:8000/redoc)** | Clean, formal OpenAPI specification viewer |
| 🩺 **System Health Status** | **[http://localhost:8000/status](http://localhost:8000/status)** | Model load verification & status JSON |

---

## 🛠️ Other Useful Commands

### 1. Re-Train the Neural Network Models
To re-run the entire data cleaning, stratified train/validation/test split, and ANN + CNN training pipeline, run the consolidated Jupyter Notebook:
```bash
# Open and run all cells in VS Code or JupyterLab:
facial_emotion.ipynb
```
Or execute it directly from the terminal via nbconvert:
```bash
jupyter nbconvert --to notebook --execute facial_emotion.ipynb
```
- **Inputs:** Raw images inside `Image_data/`
- **Pipeline:** Up to 5,000 images per class (35,000 total), grayscale/CLAHE preprocessing, stratified 80:10:10 split, ANN baseline and augmented CNN.
- **Outputs:** Saves trained models to `models/` (`ann_model.keras`, `cnn_model.keras`) and visual charts to `outputs/`.

### 2. Generate the Technical Report (PDF & HTML)
To regenerate the 24-page comprehensive project report (`report.pdf` and `report.html`):
```bash
python build_full_report.py
```

### 3. Install / Verify Dependencies
If you need to install or update dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Model Evaluation: ANN vs. CNN

The notebook measures both models on the same stratified, held-out test split, using up to 5,000 images per emotion (35,000 total). Both models train for exactly 40 epochs each. Step 6 visualizes the per-emotion training-image counts and the total available dataset-image counts. An 85% CNN accuracy is a goal, not a guarantee; reported accuracy, precision, recall, F1-score, training time, and inference latency come from the actual training and evaluation run. Results are written to `models/metrics.json` and displayed in the evaluation dashboard.

---

## 🏗️ Model Architectures

### ANN Baseline Architecture
```
Input (4,096)
  ├── Feature normalization (adapted on training data)
  ├── Dense(512) → BatchNorm → ReLU → Dropout(0.40)
  ├── Dense(256) → BatchNorm → ReLU → Dropout(0.30)
  ├── Dense(128) → BatchNorm → ReLU → Dropout(0.20)
  └── Dense(7, Softmax)
```

### CNN Architecture
```
Input (64 × 64 × 1 Grayscale Face Tensor)
  ├── Random flip, rotation and zoom during training
  ├── 2 × [Conv2D(32) → BatchNorm → ReLU] → MaxPool
  ├── 2 × [Conv2D(64) → BatchNorm → ReLU] → MaxPool
  ├── 2 × [Conv2D(128) → BatchNorm → ReLU] → MaxPool
  ├── Global Average Pooling
  ├── Dropout(0.40) → Dense(128) → Dropout(0.30)
  └── Dense(7, Softmax)
```

CNN training always runs all 40 epochs, with no early stopping. An 85–90% accuracy is a target, not a guaranteed result; report the measured validation and test accuracy for each run.

---

## 🗂️ Project Directory Structure

```
REAL TIME FACIAL EMOTION RECOGNITION/
├── Image_data/                     # Dataset (7 emotion class folders, ~49.8K images)
│   ├── angry/                      # 5,920 images
│   ├── disgust/                    # 5,920 images
│   ├── fear/                       # 5,920 images
│   ├── happy/                      # 11,398 images
│   ├── neutral/                    # 8,166 images
│   ├── sad/                        # 6,535 images
│   └── surprise/                   # 5,920 images
│
├── models/                         # Serialized model binaries & metadata
│   ├── ann_model.keras             # Trained ANN (native Keras format)
│   ├── cnn_model.keras             # Trained CNN (native Keras format)
│   ├── ann_model.pkl               # ANN metadata wrapper
│   ├── cnn_model.pkl               # CNN metadata wrapper
│   ├── preprocessing_config.pkl    # Preprocessing & label mapping config
│   ├── metrics.json                # Measured test evaluation metrics
│   └── history.json                # Epoch-by-epoch loss & accuracy history
│
├── outputs/                        # High-resolution generated charts & UI screenshots
│   ├── training_images_per_emotion.png       # Training images used per emotion
│   ├── total_dataset_images_per_emotion.png  # Total available images per emotion
│   ├── ann_history.png             # ANN accuracy and loss curves
│   ├── cnn_history.png             # CNN accuracy and loss curves
│   ├── ann_confusion_matrix.png    # ANN confusion matrix
│   ├── cnn_confusion_matrix.png    # CNN confusion matrix
│   ├── ann_vs_cnn_comparison.png   # Performance metric bar chart
│   ├── training_time_comparison.png# Wall-clock training duration chart
│   ├── sample_predictions.png      # Visual sample predictions
│   ├── ui_screenshot_home.png      # Prediction studio UI capture
│   ├── ui_screenshot_analysis.png  # Analytics dashboard UI capture
│   └── ui_screenshot_evaluation.png# Model evaluation UI capture
│
├── templates/                      # HTML5 templates (Jinja2)
│   ├── index.html                  # Home & live prediction interface
│   ├── analysis.html               # Visual analytics dashboard
│   └── evaluation.html             # Model evaluation dashboard
│
├── static/                         # Static web assets
│   ├── css/style.css               # Dark glassmorphism styling
│   └── js/script.js                # Frontend async prediction logic
│
├── venv/                           # Virtual environment
├── app.py                          # FastAPI backend application
├── facial_emotion.ipynb            # Consolidated notebook (data cleaning, training & evaluation)
├── build_full_report.py            # Generates report.html and compiles report.pdf
├── report.pdf                      # Comprehensive 24-page project report (PDF)
├── report.html                     # Standalone HTML version of the report
├── facial_emotion.ipynb            # Jupyter notebook exploration
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete project documentation
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/` | Renders Home & Live Prediction Studio |
| `GET` | `/analysis` | Renders Visual Analytics Dashboard |
| `GET` | `/evaluation` | Renders Model Evaluation Dashboard |
| `POST` | `/predict/image?model=CNN` | Predicts emotion from uploaded image file |
| `POST` | `/predict/video?model=CNN` | Processes video frames for continuous emotion detection |
| `GET` | `/metrics` | Returns stored ANN & CNN test evaluation JSON metrics |
| `GET` | `/history` | Returns epoch-by-epoch training/validation history |
| `GET` | `/analysis/data` | Returns all analytics data and plot image paths |
| `GET` | `/status` | Returns model loading status and class mapping metadata |

### Sample API Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/predict/image?model=CNN" \
     -F "file=@face_sample.jpg"
```

### Sample API JSON Response:
```json
{
  "predicted_emotion": "Happy",
  "confidence": 0.9421,
  "probabilities": {
    "Angry": 0.0012,
    "Disgust": 0.0005,
    "Fear": 0.0028,
    "Happy": 0.9421,
    "Neutral": 0.0384,
    "Sad": 0.0089,
    "Surprise": 0.0061
  },
  "inference_ms": 1.92,
  "face_detected": true,
  "model_used": "CNN"
}
```

---

## ❓ Frequently Asked Questions (FAQ) & Troubleshooting

### Q: What if PowerShell displays a script execution error when activating the venv?
Run this command in PowerShell to temporarily permit script execution for your session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Q: What if Port 8000 is already in use?
You can start Uvicorn on another port (such as `8080`):
```bash
uvicorn app:app --reload --port 8080
```
Then open `http://localhost:8080` in your browser.

### Q: Can I run inference directly from my webcam?
Yes! Click **"Start Camera"** on the Home page (`http://localhost:8000`). The frontend captures frames at configured intervals and displays real-time emotion predictions and confidence scores.

---

## 📜 License & Credits

Developed as a Deep Learning & Computer Vision project utilizing **TensorFlow**, **OpenCV**, and **FastAPI**.
- Dataset: 7-Class Facial Emotion Dataset (`Image_data`)
- Documentation Deliverable: [report.pdf](report.pdf)
