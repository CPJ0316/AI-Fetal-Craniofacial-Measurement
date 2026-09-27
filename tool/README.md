# Fetal Ultrasound Facial Measurement Tool

An AI-powered desktop tool that automatically detects the fetal facial ROI, localises eight anatomical keypoints, and computes clinical angles and distances (FMF / FMA / IFA / PL\_Dist) from fetal profile ultrasound images.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Requirements](#requirements)
3. [Model Weight Placement](#model-weight-placement)
4. [Quick Start — No IDE (for non-engineers)](#quick-start--no-ide-for-non-engineers)
5. [Developer Launch](#developer-launch)
6. [User Guide](#user-guide)
7. [Output Excel Columns](#output-excel-columns)
8. [Clinical Indicators](#clinical-indicators)
9. [Keypoint Definitions](#keypoint-definitions)
10. [FAQ](#faq)

---

## Directory Structure

```
tool/
├── main.py                  # Application entry point
├── main_window.py           # Main window logic (PyQt5)
├── main_window.ui           # Qt Designer UI layout file
├── inference_engine.py      # Inference engine (models / prediction / drawing / export)
├── README.md                # This document
│
├── models/
│   ├── detect/              # YOLO ROI detection weights (5 folds)
│   │   ├── yolo_fold1_best.pt
│   │   ├── yolo_fold2_best.pt
│   │   ├── yolo_fold3_best.pt
│   │   ├── yolo_fold4_best.pt
│   │   └── yolo_fold5_best.pt
│   │
│   └── keypoint/            # ViTPose keypoint detection weights (5 folds)
│       ├── vitpose_fold1_best.pth
│       ├── vitpose_fold2_best.pth
│       ├── vitpose_fold3_best.pth
│       ├── vitpose_fold4_best.pth
│       └── vitpose_fold5_best.pth
│
├── output/                  # Prediction Excel reports (auto-created)
├── labelme_output/          # LabelMe JSON annotations (auto-created)
└── info_image/              # Annotated info images (auto-created)
```

---

## Requirements

| Package | Recommended Version |
|---|---|
| Python | 3.9 – 3.11 |
| PyQt5 | 5.15.x |
| torch | 2.0+ (CUDA recommended) |
| torchvision | matching torch version |
| timm | 0.9+ |
| ultralytics | 8.x |
| albumentations | 1.3+ |
| opencv-python | 4.x |
| numpy | 1.x / 2.x |
| pandas | 2.x |
| openpyxl | 3.x |

It is recommended to run the tool in the same conda environment used for training:

```bash
conda activate thesis_vision
```

---

## Model Weight Placement

Model files are large and must be copied manually to the paths below.

### YOLO Detection Models

```
Source (example):
  runs/detect/yolo26n_detection_Augmentation_ver1/yolo26n_fold1/weights/best.pt
  ...  (fold 2–5 follow the same pattern)

Destination:
  tool/models/detect/yolo_fold1_best.pt
  tool/models/detect/yolo_fold2_best.pt
  tool/models/detect/yolo_fold3_best.pt
  tool/models/detect/yolo_fold4_best.pt
  tool/models/detect/yolo_fold5_best.pt
```

### ViTPose Keypoint Models

```
Source (example):
  result/Keypoint_detection/vitpose_Augmentation_new_4_IP_4_dysigma_gray/<model>/fold1/best_<model>.pth
  ...  (fold 2–5 follow the same pattern)

Destination:
  tool/models/keypoint/vitpose_fold1_best.pth
  tool/models/keypoint/vitpose_fold2_best.pth
  tool/models/keypoint/vitpose_fold3_best.pth
  tool/models/keypoint/vitpose_fold4_best.pth
  tool/models/keypoint/vitpose_fold5_best.pth
```

---

## Quick Start — No IDE (for non-engineers)

This section explains how to install the environment and launch the tool using only **Windows Explorer** and the **Command Prompt**, without any programming IDE.

### Step 1 — Install Miniconda (one-time setup)

1. Open your web browser and go to:
   **https://docs.anaconda.com/miniconda/**
2. Download the **Windows 64-bit** installer (`.exe` file).
3. Double-click the downloaded file and follow the on-screen instructions.
   - When asked "Add Miniconda to PATH", choose **Yes** (or tick the checkbox).
4. After installation, open the **Start Menu**, search for **"Anaconda Prompt"**, and open it.
   You should see a black window with `(base)` at the start of the line.

### Step 2 — Create the Python environment (one-time setup)

In the Anaconda Prompt window, type the following commands one by one and press **Enter** after each:

```
conda create -n fetal_tool python=3.10 -y
conda activate fetal_tool
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install PyQt5 timm ultralytics albumentations opencv-python pandas openpyxl
```

> **Note:** If your computer does **not** have an NVIDIA GPU, replace the `pip install torch` line with:
> ```
> pip install torch torchvision
> ```

Wait until all packages finish downloading and installing (this may take several minutes).

### Step 3 — Place the model weight files

1. Open **Windows Explorer** and navigate to the `tool\models\detect\` folder.
2. Copy the five YOLO weight files (`yolo_fold1_best.pt` … `yolo_fold5_best.pt`) into this folder.
3. Navigate to `tool\models\keypoint\` and copy the five ViTPose weight files
   (`vitpose_fold1_best.pth` … `vitpose_fold5_best.pth`) into this folder.

The folder should look like this after copying:

```
tool/
└── models/
    ├── detect/
    │   ├── yolo_fold1_best.pt
    │   ├── yolo_fold2_best.pt
    │   ├── yolo_fold3_best.pt
    │   ├── yolo_fold4_best.pt
    │   └── yolo_fold5_best.pt
    └── keypoint/
        ├── vitpose_fold1_best.pth
        ├── vitpose_fold2_best.pth
        ├── vitpose_fold3_best.pth
        ├── vitpose_fold4_best.pth
        └── vitpose_fold5_best.pth
```

### Step 4 — Launch the tool

1. Open **Anaconda Prompt** again (search "Anaconda Prompt" in the Start Menu).
2. Type the following commands and press **Enter** after each:

```
conda activate fetal_tool
cd /d "E:\CPJ\thesis_ver2\tool"
python main.py
```

> Replace `E:\CPJ\thesis_ver2\tool` with the actual path to your `tool` folder if it is different.
> You can find the correct path by right-clicking the `tool` folder in Windows Explorer
> and selecting **"Copy as path"**, then paste it after `cd /d `.

3. The application window will open. You are ready to use the tool.

### Step 5 — Launching the tool next time

Every subsequent launch only requires steps 4.1–4.3 above (you do **not** need to reinstall anything).
For convenience you can create a shortcut:

1. Right-click on your Desktop → **New → Shortcut**.
2. In the "Type the location" box, enter:
   ```
   cmd /k "conda activate fetal_tool && cd /d E:\CPJ\thesis_ver2\tool && python main.py"
   ```
3. Click **Next**, give it a name such as **"Fetal Measurement Tool"**, and click **Finish**.
4. Double-click the shortcut icon to launch the tool directly.

---

## Developer Launch

```bash
conda activate thesis_vision
cd tool
python main.py
```

---

## User Guide

### 1. Open file / folder

Click **"Open file/folder"**. A dialog will ask how to load images:

- **Yes** → pick a single image file (`.png / .jpg / .jpeg / .bmp / .tiff`).
  If an unsupported format is selected, the message `File/folder load in fail, please select again.` is shown and the dialog reopens.
- **No** → pick a folder; all supported images inside are loaded automatically.
  If no images are found, the same error message is shown.

---

### 2. Predict

Click **"Predict"** to run the full pipeline on all loaded images:

1. **YOLO 5-Fold Ensemble** — detects the facial ROI; a bounding box is accepted only when ≥ 3 folds agree (majority vote).
2. **ViTPose 5-Fold + TTA** — detects 8 facial keypoints on the cropped ROI at three scales (512 / 640 / 768 px); results are averaged across folds and scales.
3. **Clinical indicator calculation** — computes FMF, FMA, IFA, and PL\_Dist.

When all images are done:
- A dialog shows **"All images are predicted."**
- A timestamped Excel file (e.g. `prediction_20260920_143022.xlsx`) is saved to the `output/` folder.
  Each click of **Predict** generates a new file.

---

### 3. Browse results

#### Image view (left — graphicsView)

Displays the current image with overlaid:
- **Orange rectangle** — ROI bounding box
- **Blue dots** — 8 detected keypoints
- **Blue extended lines** — forehead line, palate line, face profile line, nasion normal line, PL baseline
- **Green text** — FMF / FMA / IFA / PL\_Dist values

#### Measurement panel (bottom — textBrowser)

Shows the measurement values and diagnostic suggestions for the current image.

#### Switch between images

| Key | Action |
|---|---|
| `D` | Next image |
| `A` | Previous image |

- Reached the first image: `Already at the first image.`
- Reached the last image: `Already at the last image.`

---

### 4. Output LabelMe JSON

Click **"Output LabelMe file"** to export predictions as [LabelMe](https://github.com/labelmeai/labelme)-compatible `.json` files to `labelme_output/`.

Each JSON contains:

| Label | Shape type | Description |
|---|---|---|
| `able_ROI` | rectangle | ROI bounding box |
| `forehead_1` … `mentum_tip` | point | 8 keypoints |
| `FMF_1`, `FMF_2` | line | Forehead and palate lines |
| `FMA_2` | line | Face profile line |
| `IFA_2` | line | Nasion perpendicular normal |
| `PL_distance_1` | line | PL baseline (nasion – mandible) |
| `PL_distance` | line | PL perpendicular projection |

---

### 5. Output info image

Click **"Output image with information"** to save one composite image per input to `info_image/` (filename: `<original_name>_info.png`).

Layout:
- **Left** — annotated prediction image
- **Right** — white panel with measurement values (with units) and diagnostic suggestions

---

## Output Excel Columns

| Column | Description |
|---|---|
| Index | Image sequence number |
| Image\_Name | Filename |
| Image\_Path | Full file path |
| ROI\_x1/y1/x2/y2 | Detected ROI coordinates (pixels) |
| Det\_Success | Whether ROI was detected |
| forehead\_1\_x/y … mentum\_tip\_x/y | 8 keypoint pixel coordinates |
| FMF / FMA / IFA | Clinical angles (degrees) |
| PL\_Dist | Profile line distance (pixels) |
| Diagnosis | Diagnostic suggestion text |

---

## Clinical Indicators

| Indicator | Full Name | Normal Range | Unit | Clinical Significance |
|---|---|---|---|---|
| **FMF** | Frontomaxillary Facial Angle | 75° – 85° | deg | Angle between forehead and palate lines; low values may indicate Trisomy 21 |
| **FMA** | Fetal Mandibulo-Facial Angle | 60° – 75° | deg | Angle between palate and face profile lines; abnormal values may indicate micrognathia |
| **IFA** | Inferior Facial Angle | 50° – 65° | deg | Angle between the nasion normal and face profile line; low values warrant chromosomal workup |
| **PL\_Dist** | Profile Line Distance | 0 – 15 | px | Perpendicular distance from forehead\_1 to the nasion–mandible line |

> **Disclaimer:** This tool is intended as a measurement aid only. All diagnostic suggestions must be reviewed and confirmed by qualified medical professionals. The outputs do not constitute a clinical diagnosis.

---

## Keypoint Definitions

| Index | Name | Anatomical Location |
|---|---|---|
| 0 | forehead\_1 | Frontal bone endpoint 1 |
| 1 | forehead\_2 | Frontal bone endpoint 2 |
| 2 | nasion | Nasion (root of nose) |
| 3 | palate\_1 | Hard palate endpoint 1 |
| 4 | palate\_2 | Hard palate endpoint 2 |
| 5 | upper\_lip | Upper lip |
| 6 | mandible | Mandible |
| 7 | mentum\_tip | Tip of the chin (mentum) |

---

## FAQ

**Q: The application shows "model files are missing" on startup.**
A: Follow the [Model Weight Placement](#model-weight-placement) section to copy the weights to the correct paths, then restart.

**Q: Prediction is very slow.**
A: An NVIDIA GPU is strongly recommended. Verify CUDA is available by running `python -c "import torch; print(torch.cuda.is_available())"` in the Anaconda Prompt — it should print `True`.

**Q: ROI detection failed (Det\_Success = False).**
A: The image may be of insufficient quality or the viewing angle may differ from the training distribution. Ensure the image shows a standard fetal profile plane.

**Q: Some keypoints show N/A.**
A: A keypoint is marked invalid when fewer than 3 out of 5 folds agree on its location. The corresponding clinical indicator will also show N/A.

**Q: LabelMe cannot display the image after opening the JSON.**
A: Place the `.json` file and the corresponding image file in the same folder, or manually specify the image path inside LabelMe.

**Q: The application window does not open and the Anaconda Prompt shows an error.**
A: Make sure the `fetal_tool` conda environment is activated (`conda activate fetal_tool`) and that all packages were installed successfully. Copy the full error message and contact the developer.
