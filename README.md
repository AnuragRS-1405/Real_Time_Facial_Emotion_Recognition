# Real-Time Facial Emotion Recognition
### Deep Learning System Powered by Deep ANN & 4-Stage Hierarchical CNN with FastAPI & Glassmorphism UI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📋 Project Overview

This project is an **end-to-end Affective Computing & Deep Learning system** engineered to detect and classify human facial expressions in real time from static images, uploaded files, and live webcam feeds.

The system classifies human emotional states into **7 canonical emotions**:

| Emotion | Label | Visual Features & Morphological Markers |
|:---|:---:|:---|
| 😠 **Angry** | `angry` | Knitted eyebrows, lowered brow furrow, intense stare, tightened lips |
| 🤢 **Disgust** | `disgust` | Wrinkled nose bridge, raised upper lip, narrowed eyes |
| 😨 **Fear** | `fear` | Widened eye aperture, raised inner eyebrows, parted lips |
| 😊 **Happy** | `happy` | Elevated zygomatic cheek muscles, broad smile, crinkled eye corners |
| 😐 **Neutral** | `neutral` | Relaxed facial musculature, horizontal lip line, resting gaze |
| 😢 **Sad** | `sad` | Downturned mouth corners, inner eyebrows pulled upwards, drooping eyelids |
| 😲 **Surprise** | `surprise` | Vertically dropped jaw, wide circular eye aperture, arched eyebrows |

---

## 🔬 Core Scientific & Engineering Highlights

1. **Balanced Dataset & MD5 Deduplication Pipeline:**
   - Ingests up to **5,000 images per emotion class** (up to **35,000 images total**) from `Image_data/` to eliminate severe class imbalance.
   - Computes **MD5 hash digests** on every image to remove exact duplicates, ensuring zero data leakage between training, validation, and test splits.
   - Applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** with `clipLimit=2.0` and `tileGridSize=(8, 8)` to normalize ambient lighting and contrast across faces.
   - Deterministic resizing to $64 \times 64$ single-channel grayscale with float32 scaling into $[0.0, 1.0]$.
   - Stratified dataset partitioning: **80% Training**, **10% Validation**, and **10% Held-Out Testing** (`random_state=42`).

2. **Dual-Model Architectural Comparison:**
   - **Deep Baseline ANN (Multi-Layer Perceptron):**
     - Flattens the $64 \times 64$ face into a 4,096-dimensional 1D vector.
     - Adapts a Keras `Normalization` layer over training features.
     - Dense hierarchy: $512 \rightarrow 256 \rightarrow 128 \rightarrow 7$ with He-normal weight initialization, Batch Normalization, Dropout ($0.40 \rightarrow 0.30 \rightarrow 0.20$), and $L_2$ weight regularization ($1\times 10^{-4}$).
     - Benchmarks performance when 2D spatial locality is discarded.
   - **Improved 4-Stage Hierarchical CNN:**
     - Online data augmentation: `RandomFlip("horizontal")`, `RandomRotation(0.08)`, `RandomZoom(0.10)`, and `RandomTranslation(0.08, 0.08)`.
     - 4 hierarchical convolutional blocks: $32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$ filters with $3 \times 3$ receptive fields, He-normal initialization, Batch Normalization, and staged Dropout ($0.10 \rightarrow 0.15 \rightarrow 0.20$).
     - Preserves spatial geometry into a $4 \times 4$ layout (avoiding premature global pooling) into a regularized dense head (Dense 256 $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout 0.30 $\rightarrow$ Dense 7 Softmax).
     - Trained with **Categorical Cross-Entropy with Label Smoothing ($0.1$)** to avoid overconfident misclassifications, `ReduceLROnPlateau` (decay factor 0.5), and `EarlyStopping` (restoring best validation weights).

3. **Inference with Flip Test-Time Augmentation (TTA):**
   - At test time and in production serving (`app.py`), the CNN evaluates both the original face and its horizontal mirror:
     $$\hat{P} = \frac{P(\mathbf{x}) + P(\mathbf{x}_{\text{flipped}})}{2}$$
   - Ensembling the two views cancels pose asymmetries and boosts generalization on real-world faces.

4. **Real-Time Computer Vision & Face Detection:**
   - Uses OpenCV's **Haar Feature-Based Cascade Classifier** (`haarcascade_frontalface_default.xml`) to detect and crop primary face ROIs in $\mathcal{O}(1)$ time via integral images.
   - Automatic fallback to full-frame preprocessing if the face is already cropped or partially obscured.

5. **Production FastAPI Backend with Auto-Artifact Refreshing:**
   - Asynchronous FastAPI server (`app.py`) with CORS middleware and OpenAPI Swagger/ReDoc.
   - Features `refresh_artifacts()` to automatically load newly trained `.keras` models, `metrics.json`, and `history.json` on-demand without requiring a server reboot.

6. **Responsive Dark Glassmorphism Frontend:**
   - **Home / Prediction Studio (`/`):** Live image upload, drag-and-drop, real-time webcam stream prediction, model toggle (ANN vs. CNN), and 7-class probability meters.
   - **Visual Analytics (`/analysis`):** Dataset class distributions, training curves, confusion matrices, and qualitative sample predictions.
   - **Model Evaluation (`/evaluation`):** Side-by-side metric comparison cards (Accuracy, Precision, Recall, F1, Latency), per-class F1 canvas charts, and architectural conclusions.

---

## ⚡ Execution Commands & Running the Project

Follow these steps to launch the system locally.

### Step 1: Open Your Terminal
Navigate to the project root directory:
```bash
cd "c:\Users\Anurag Rajendra\OneDrive\Pictures\PROJECTS\REAL TIME FACIAL EMOTION RECOGNITION"
```

---

### Step 2: Activate the Virtual Environment

The project includes a configured virtual environment inside `venv/`. Activate it based on your shell:

#### 🔹 Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
> **PowerShell Execution Policy Tip:** If you encounter `Execution of scripts is disabled on this system`, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

#### 🔹 Windows Command Prompt (CMD):
```cmd
venv\Scripts\activate.bat
```

#### 🔹 macOS / Linux:
```bash
source venv/bin/activate
```

*(Once activated, `(venv)` will precede your terminal prompt).*

---

### Step 3: Run the FastAPI Web Server

You can run the web application using any of the following methods:

#### Option A: Direct Uvicorn Command (Recommended)
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

#### Option B: Direct Python Execution
```bash
python app.py
```

#### Option C: One-Line Direct Launch (Without Manual Activation)
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
| ⚖️ **Model Evaluation** | **[http://localhost:8000/evaluation](http://localhost:8000/evaluation)** | Measured test metrics, per-class F1 & comparison tables |
| 📖 **Interactive Swagger Docs** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | Test the REST API endpoints directly via interactive UI |
| 📑 **ReDoc Documentation** | **[http://localhost:8000/redoc](http://localhost:8000/redoc)** | Clean, formal OpenAPI specification documentation |
| 🩺 **System Health Status** | **[http://localhost:8000/status](http://localhost:8000/status)** | Model load verification & metadata JSON |

---

## 📓 Notebook Pipeline: `facial_emotion.ipynb`

All data preprocessing, deduplication, exploratory data analysis, and neural network training are structured into **18 transparent steps** within [`facial_emotion.ipynb`](facial_emotion.ipynb):

1. **Step 1: Import Packages & Set Seeds:** Imports TensorFlow, Keras, OpenCV, Seaborn, and configures frontend chart styling (`style_chart_for_frontend`).
2. **Step 2: Folders & Class Names:** Initializes paths, classes, and constants ($64 \times 64$, batch size 64).
3. **Step 3: Balanced Image Search:** Collects up to 5,000 files per emotion class from `Image_data/` (up to 35,000 images).
4. **Step 4: Preprocessing & Deduplication:** Grayscale conversion, $64 \times 64$ resize, MD5 hash duplicate elimination, CLAHE illumination balancing, and normalization to $[0.0, 1.0]$.
5. **Step 5: Stratified Data Splitting:** 80% train, 10% validation, 10% held-out test splits.
6. **Step 6: Class Distribution Charts:** Saves `training_images_per_emotion.png` and `total_dataset_images_per_emotion.png` to `outputs/`.
7. **Step 7: Build Baseline ANN:** 4,096-input MLP with adapted normalization, $512 \rightarrow 256 \rightarrow 128$ dense blocks, BatchNorm, and Dropout.
8. **Step 8: Train Baseline ANN:** Trains for up to 40 epochs with `ReduceLROnPlateau` and `EarlyStopping`.
9. **Step 9: Build Improved 4-Stage CNN:** Hierarchical $32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$ ConvNet with online augmentation, spatial flattening, and label smoothing.
10. **Step 10: Train Improved CNN:** Trains for up to 60 epochs with `ReduceLROnPlateau` and `EarlyStopping` restoring best validation weights.
11. **Step 11: Prediction & Flip TTA:** Evaluates ANN and CNN with Flip Test-Time Augmentation on untouched test images.
12. **Step 12: Test-Set Evaluation:** Generates full classification reports (Accuracy, Precision, Recall, F1).
13. **Step 13: Export Trained Models:** Serializes `ann_model.keras` and `cnn_model.keras` to `models/`.
14. **Step 14: Save Training Charts:** Exports `ann_history.png` and `cnn_history.png` to `outputs/`.
15. **Step 15: Save Confusion Matrices:** Exports `ann_confusion_matrix.png` and `cnn_confusion_matrix.png` to `outputs/`.
16. **Step 16: Comparative Visualizations:** Saves `ann_vs_cnn_comparison.png` and `training_time_comparison.png` to `outputs/`.
17. **Step 17: Visual Sample Predictions:** Exports `sample_predictions.png` showcasing qualitative predictions for each emotion.
18. **Step 18: Save Production Metadata:** Exports `metrics.json`, `history.json`, and `preprocessing_config.pkl` to `models/`.

To re-run the notebook from the terminal:
```bash
jupyter nbconvert --to notebook --execute facial_emotion.ipynb
```

---

## 🏗️ Neural Network Architectures

### 1. Deep Baseline ANN Architecture
```
Input Face Vector (4,096-dimensional flattened 64 × 64 pixels)
  │
  ├── Keras Normalization Layer (Adapted on Training Data)
  │
  ├── Block 1: Dense(512, HeNormal, L2=1e-4) ──> BatchNorm ──> ReLU ──> Dropout(0.40)
  │
  ├── Block 2: Dense(256, HeNormal, L2=1e-4) ──> BatchNorm ──> ReLU ──> Dropout(0.30)
  │
  ├── Block 3: Dense(128, HeNormal, L2=1e-4) ──> BatchNorm ──> ReLU ──> Dropout(0.20)
  │
  └── Output Layer: Dense(7, Softmax) ──> [Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise]
```

### 2. Improved 4-Stage Hierarchical CNN Architecture
```
Input Face Tensor (64 × 64 × 1 Grayscale)
  │
  ├── Online Data Augmentation (RandomFlip, RandomRotation ±8°, RandomZoom ±10%, RandomTranslation ±8%)
  │
  ├── Block 1 (64×64 → 32×32):
  │     Conv2D(32, 3×3, Same, HeNormal) ──> BatchNorm ──> ReLU
  │     Conv2D(64, 3×3, Same, HeNormal) ──> BatchNorm ──> ReLU ──> MaxPool(2×2) ──> Dropout(0.10)
  │
  ├── Block 2 (32×32 → 16×16):
  │     Conv2D(128, 3×3, Same, HeNormal) ──> BatchNorm ──> ReLU
  │     Conv2D(128, 3×3, Same, HeNormal) ──> BatchNorm ──> ReLU ──> MaxPool(2×2) ──> Dropout(0.15)
  │
  ├── Block 3 (16×16 → 8×8):
  │     Conv2D(256, 3×3, Same, HeNormal) ──> BatchNorm ──> ReLU
  │     Conv2D(256, 3×3, Same, HeNormal) ──> BatchNorm ──> ReLU ──> MaxPool(2×2) ──> Dropout(0.20)
  │
  ├── Block 4 (8×8 → 4×4):
  │     Conv2D(512, 3×3, Same, HeNormal) ──> BatchNorm ──> ReLU ──> MaxPool(2×2)
  │
  ├── Spatial Feature Preservation (4×4 spatial layout maintained):
  │     Flatten() ──> Dropout(0.40)
  │
  ├── Regularized Dense Head:
  │     Dense(256, HeNormal) ──> BatchNorm ──> ReLU ──> Dropout(0.30)
  │
  └── Output Layer: Dense(7, Softmax) ──> Categorical Cross-Entropy (Label Smoothing = 0.1)
```

---

## 🗂️ Project Directory Structure

```
REAL TIME FACIAL EMOTION RECOGNITION/
├── Image_data/                             # Dataset (7 emotion folders, ~49.8K raw images)
│   ├── angry/                              # Angry facial images
│   ├── disgust/                            # Disgust facial images
│   ├── fear/                               # Fear facial images
│   ├── happy/                              # Happy facial images
│   ├── neutral/                            # Neutral facial images
│   ├── sad/                                # Sad facial images
│   └── surprise/                           # Surprise facial images
│
├── models/                                 # Serialized model weights & production metadata
│   ├── ann_model.keras                     # Trained Baseline ANN (native Keras format)
│   ├── cnn_model.keras                     # Trained Improved CNN (native Keras format)
│   ├── preprocessing_config.pkl            # Preprocessing, label mapping & normalization metadata
│   ├── metrics.json                        # Measured test-set evaluation metrics
│   └── history.json                        # Epoch-by-epoch loss & accuracy history
│
├── outputs/                                # Visual analytics charts & UI captures
│   ├── training_images_per_emotion.png     # Step 6: Training images used per emotion
│   ├── total_dataset_images_per_emotion.png# Step 6: Total available images per emotion
│   ├── ann_history.png                     # Step 14: ANN training/validation accuracy & loss curves
│   ├── cnn_history.png                     # Step 14: CNN training/validation accuracy & loss curves
│   ├── ann_confusion_matrix.png            # Step 15: ANN confusion matrix heatmap
│   ├── cnn_confusion_matrix.png            # Step 15: CNN confusion matrix heatmap
│   ├── ann_vs_cnn_comparison.png           # Step 16: Performance metric comparison bar chart
│   ├── training_time_comparison.png        # Step 16: Wall-clock training duration chart
│   ├── sample_predictions.png              # Step 17: Qualitative test sample predictions
│   ├── ui_screenshot_home.png              # Home viewport capture
│   ├── ui_screenshot_analysis.png          # Visual analytics dashboard capture
│   └── ui_screenshot_evaluation.png        # Model evaluation dashboard capture
│
├── templates/                              # Jinja2 HTML5 front-end templates
│   ├── index.html                          # Prediction Studio & live detection
│   ├── analysis.html                       # Visual analytics dashboard
│   └── evaluation.html                     # Model comparison & performance metrics dashboard
│
├── static/                                 # Static assets
│   ├── css/style.css                       # Premium dark glassmorphism styling
│   └── js/script.js                        # Client-side async prediction & Canvas charting logic
│
├── app.py                                  # FastAPI asynchronous REST backend
├── facial_emotion.ipynb                    # 18-step consolidated research & training notebook
├── build_full_report.py                    # Compiles report.html and generates report.pdf
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/` | Serves the Home & Live Prediction Studio UI |
| `GET` | `/analysis` | Serves the Visual Analytics Dashboard UI |
| `GET` | `/evaluation` | Serves the Model Evaluation & Benchmarking UI |
| `POST` | `/predict/image?model=CNN` | Classifies emotion from uploaded face image using ANN or CNN (with Flip TTA) |
| `GET` | `/metrics` | Returns stored test-set JSON evaluation metrics |
| `GET` | `/history` | Returns epoch-by-epoch training and validation metrics |
| `GET` | `/analysis/data` | Returns all analytics metadata, class lists, and plot image paths |
| `GET` | `/status` | Returns system health, model load status, and preprocessing metadata |

### Sample Inference Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/predict/image?model=CNN" \
     -F "file=@sample_face.jpg"
```

### Sample Inference JSON Response:
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
  "inference_ms": 2.45,
  "face_detected": true,
  "model_used": "CNN"
}
```

---

## ❓ Frequently Asked Questions (FAQ) & Troubleshooting

### Q1: What if PowerShell displays a script execution policy error?
Run this command in your PowerShell window to permit local script execution:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Q2: What if Port 8000 is already in use by another service?
Launch Uvicorn on an alternate port (such as `8080`):
```bash
uvicorn app:app --reload --port 8080
```
Then open `http://localhost:8080` in your browser.

### Q3: Do I need to restart `app.py` after re-running `facial_emotion.ipynb`?
No. The FastAPI backend includes a dynamic `refresh_artifacts()` routine that automatically detects and reloads `ann_model.keras`, `cnn_model.keras`, `metrics.json`, and `history.json` on the next incoming request without needing a manual server restart.

### Q4: How does Flip Test-Time Augmentation (TTA) work in inference?
When `model=CNN` is selected, `app.py` evaluates both the cropped face $\mathbf{x}$ and its horizontal mirror $\mathbf{x}_{\text{flipped}}$. The two softmax probability distributions are averaged before identifying the argmax emotion. This eliminates unilateral expression bias (e.g., asymmetric half-smiles).

---

## 📜 License & Credits

Developed as an Affective Computing & Computer Vision project utilizing **TensorFlow**, **OpenCV**, and **FastAPI**.
- **Dataset:** 7-Class Facial Emotion Dataset (`Image_data/`)
- **License:** [MIT License](LICENSE)
