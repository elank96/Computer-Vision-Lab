# Computer Vision Lab — Technical Architecture & Implementation

## 1. Executive Summary

This project implements **video anomaly detection** using open-vocabulary object detection. It targets explosion/fire content in video by:

1. **Ingesting** a video and ground-truth anomaly frame ranges from an annotation file.
2. **Processing** each frame with **GroundingDINO** to detect text-prompted concepts (e.g., "smoke or fire").
3. **Evaluating** predictions against ground truth via precision, recall, and accuracy.

The codebase is organized as Jupyter notebooks for Colab-style execution, with dependencies on OpenCV, PyTorch, the GroundingDINO repository, and the supervision library.

---

## 2. Application Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTER VISION LAB PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐  │
│  │   Data       │    │  Video Frame     │    │  Open-Vocabulary         │  │
│  │   Ingestion  │───▶│  Extraction      │───▶│  Detection (GroundingDINO)│  │
│  │              │    │  (OpenCV)        │    │                          │  │
│  └──────────────┘    └──────────────────┘    └────────────┬────────────┘  │
│         │                          │                       │               │
│         │                          │                       ▼               │
│         │                          │              ┌─────────────────────┐  │
│         │                          │              │  Anomaly Frame      │  │
│         │                          │              │  Prediction List    │  │
│         │                          │              └──────────┬──────────┘  │
│         │                          │                         │             │
│         ▼                          ▼                         ▼             │
│  ┌──────────────┐          ┌──────────────┐         ┌─────────────────┐  │
│  │ annotation   │          │ Ground Truth  │         │  Evaluation      │  │
│  │ .txt         │          │ Boolean Mask │         │  (TP/FP/TN/FN,   │  │
│  │ (ranges)     │          │ (per frame)  │         │   P, R, Acc)     │  │
│  └──────┬───────┘          └──────┬───────┘         └─────────────────┘  │
│         │                         │                         ▲             │
│         └─────────────────────────┴─────────────────────────┘             │
│                           Comparison & Metrics                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Role | Primary Artifacts |
|-----------|------|-------------------|
| **Data ingestion** | Video path + annotation file (video name, label, frame ranges) | `annotation.txt`, video file (e.g. `Explosion033_x264.mp4`) |
| **Video I/O** | Frame-by-frame read and metadata (count, size) | `cv2.VideoCapture`, frame index, image arrays |
| **Ground-truth derivation** | Map annotation ranges → per-frame boolean list | `get_anomalous_frames()`, `generate_anomaly_boolean_list()` |
| **Detection** | Load GroundingDINO, run per-frame inference with text prompt | `load_model`, `load_image`, `predict`, `annotate` |
| **Evaluation** | Compare predicted vs. ground-truth anomaly frames | TP/FP/TN/FN rates, precision, recall, accuracy |

### 2.3 Repository Layout

```
Computer-Vision-Lab/
├── annotation.txt                    # Ground-truth: video name, label, frame ranges
├── video_frame_extractor.ipynb       # Video I/O, annotation parsing, frame viz
├── GroundingDINO_Video_Frame_Analysis.ipynb   # Full pipeline: DINO + evaluation
├── GroundingDINO.ipynb               # Standalone GroundingDINO setup/demos
├── TECHNICAL_ARCHITECTURE.md         # This document
└── videos/                           # (Expected) Video assets, e.g. Explosion033_x264.mp4
```

---

## 3. Data Model & Annotation Format

### 3.1 Annotation File (`annotation.txt`)

Single-line, space-separated format:

```
<video_filename> <label> <start_1> <end_1> [<start_2> <end_2> ...]
```

**Example:**

```
Explosion033_x264.mp4  Explosion  970  1350  1550  3153
```

- **Video name:** `Explosion033_x264.mp4`
- **Label:** `Explosion`
- **Anomaly ranges (inclusive frame indices):** `(970, 1350)` and `(1550, 3153)`

Frames inside any `(start, end)` pair are considered anomalous; all other frames are normal.

### 3.2 Derived Structures

- **`anomaly_ranges`:** List of `(start, end)` tuples from `get_anomalous_frames()`.
- **`anomaly_boolean_list`:** Length = number of frames; `True` iff that frame index lies in any anomaly range.
- **`predicted_anomaly_frames`:** List of frame indices where the model detected the target concept (e.g., "smoke or fire").

---

## 4. Implementation Flow

### 4.1 Video Frame Extractor Notebook (`video_frame_extractor.ipynb`)

**Purpose:** Setup, video inspection, annotation parsing, and optional frame visualization.

1. **Environment:** Install `opencv-python`, `matplotlib`, `gitpython`; clone repo; set `explosionVideo` path.
2. **Video metadata:** `cv2.VideoCapture(video_path)`; read `CAP_PROP_FRAME_COUNT`, frame height/width (e.g. 3154 frames, 240×320).
3. **Display:** Optional in-notebook playback via `IPython.display.Video`.
4. **Annotation parsing:**
   - `get_anomalous_frames(filename)` reads one line, splits, and builds `[(start1,end1), (start2,end2), ...]`.
   - `generate_anomaly_boolean_list(video_path, anomaly_ranges)` iterates all frames and sets a boolean per frame from the ranges.
5. **Visualization:** 7×5 grid of frames sampled every 100 frames (`frame % 100 == 0`), shown with matplotlib (`cv2.cvtColor(..., COLOR_BGR2RGB)`).

This notebook does **not** run a detector; it only prepares data and labels.

### 4.2 GroundingDINO Video Frame Analysis Notebook (`GroundingDINO_Video_Frame_Analysis.ipynb`)

**Purpose:** End-to-end anomaly detection and evaluation using GroundingDINO.

**Flow:**

1. **Setup**
   - Clone project repo; set `explosionVideo` and `annotation_file`.
   - Parse ground truth: `get_anomalous_frames(annotation_file)` → `anomaly_ranges`; `generate_anomaly_boolean_list(...)` → `anomaly_boolean_list`.

2. **GroundingDINO environment**
   - Set `HOME`; clone [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO); `pip install -e .` and `roboflow`.
   - Config: `GroundingDINO_SwinT_OGC.py`.
   - Weights: download `groundingdino_swint_ogc.pth` into `HOME/weights`.
   - Load model: `load_model(CONFIG_PATH, WEIGHTS_PATH)` (uses BERT for text and Swin-T backbone).

3. **Per-frame detection loop**
   - Open video with `cv2.VideoCapture(explosionVideo)`; get `n_frames`.
   - For each frame:
     - Read frame; write to a fixed path (e.g. `videos/frame.jpg`).
     - `load_image(IMAGE_PATH)` → `image_source`, `image`.
     - `predict(model, image, caption=TEXT_PROMPT, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD)` → `boxes`, `logits`, `phrases`.
     - If `len(phrases) > 0`, append current frame index to `predicted_anomaly_frames`.
     - Optionally `annotate(...)` and display with `supervision.plot_image()`.

   **Parameters used:**  
   - `TEXT_PROMPT = "smoke or fire"`  
   - `BOX_TRESHOLD = 0.35`  
   - `TEXT_TRESHOLD = 0.25`

4. **Evaluation**
   - **Counts:** `num_anomalies` = sum of lengths of anomaly ranges; `num_non_anomalies` = `n_frames - num_anomalies`.
   - **Rates (fractions, not counts):**
     - **True Positive (TP):** fraction of anomaly frames that were predicted as anomaly.
     - **False Positive (FP):** fraction of non-anomaly frames predicted as anomaly.
     - **True Negative (TN):** fraction of non-anomaly frames correctly predicted as non-anomaly.
     - **False Negative (FN):** fraction of anomaly frames missed.
   - **Metrics:**
     - Precision = TP / (TP + FP)
     - Recall = TP / (TP + FN)
     - Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Reported example (explosion video):**  
TP ≈ 0.55, FP ≈ 0.06, TN ≈ 0.94, FN ≈ 0.45 → Precision ≈ 0.91, Recall ≈ 0.55, Accuracy ≈ 0.74.

### 4.3 GroundingDINO Notebook (`GroundingDINO.ipynb`)

**Purpose:** Isolated GroundingDINO setup and image-level demos (no video loop). It clones the repo, downloads weights and sample images (e.g. dog images), loads the model, and runs `predict` + `annotate` on single images with various text prompts. Used as a reference/tutorial for the detector rather than as part of the anomaly pipeline.

---

## 5. Machine Learning Concepts & Architecture

### 5.1 Problem Formulation

- **Task:** Binary classification per frame — anomaly (e.g. explosion/fire/smoke) vs. normal.
- **Approach:** Open-vocabulary detection: a single text prompt ("smoke or fire") defines the “anomaly” concept; any detection above thresholds is treated as a positive for that frame.
- **Training:** The project uses a **pre-trained** GroundingDINO model; no fine-tuning is performed in these notebooks.

### 5.2 GroundingDINO in Short

- **Goal:** Detect objects described by free-form text (open vocabulary), not a fixed set of classes.
- **Core idea:** Fuse vision and language so that image regions are scored against text embeddings; the best-matching regions become detections.

**Main architectural components:**

1. **Backbone (vision):** Swin Transformer (Swin-T in config `GroundingDINO_SwinT_OGC.py`) to get multi-scale image features.
2. **Text encoder:** BERT (e.g. `bert-base-uncased`) to encode the prompt into token embeddings.
3. **Feature enhancer:** Early fusion of image and text via deformable self-attention and cross-attention, producing a shared grounded feature space.
4. **Language-guided query selection:** Queries are initialized from image regions that score highly against the text, instead of fixed learned queries.
5. **Cross-modality decoder:** Alternating self-attention on queries and image–text / text–image cross-attention to refine boxes and align with the text.
6. **Sub-sentence grounding:** Phrases (e.g. “smoke or fire”) can be encoded with block-diagonal attention to keep multi-word concepts coherent.

**Inference parameters:**

- **Box threshold:** Minimum score for a box to be kept (e.g. 0.35).
- **Text threshold:** Minimum alignment between box and text (e.g. 0.25).  
Lower values allow more detections; higher values make the detector more conservative.

### 5.3 Why This Fits Anomaly Detection

- Anomalies (fire, smoke, explosion) are **semantic concepts** that can be described in language.
- GroundingDINO avoids training a fixed set of anomaly classes; the same pipeline can be retargeted by changing the text prompt (e.g. “person”, “weapon”, “damage”).
- Per-frame application treats each frame as an image; temporal modeling (e.g. across frames) is not used — detection is **frame-independent**.

### 5.4 Evaluation Metrics (ML Perspective)

- **Precision:** Among frames predicted as anomaly, how many are truly anomaly (reduces false alarms).
- **Recall:** Among true anomaly frames, how many are predicted (reduces missed events).
- **Accuracy:** Overall correct decisions over all frames (can be dominated by non-anomaly frames if the video is mostly normal).
- **Trade-off:** High precision (0.91) with moderate recall (0.55) in the reported run suggests the model is conservative: fewer false positives but misses about half of the anomaly frames. Tuning box/text thresholds can shift this trade-off.

---

## 6. Dependencies & Execution Environment

- **Python:** Used with Colab (e.g. Python 3.10).
- **Libraries:**  
  - `opencv-python` (cv2) — video I/O and image write.  
  - `torch` — GroundingDINO.  
  - `supervision` — visualization (`plot_image`).  
  - GroundingDINO repo (install via `pip install -e .`).  
  - Optional: `roboflow`, `matplotlib`, `pandas`, `numpy`, `tqdm`, `gitpython`.
- **Hardware:** GPU (e.g. Tesla T4) recommended for GroundingDINO; `nvidia-smi` is run in the notebooks.
- **External assets:**  
  - Annotation: local `annotation.txt`.  
  - Video: path set in notebook (e.g. `explosionVideo`).  
  - Model weights: downloaded from GitHub releases (`groundingdino_swint_ogc.pth`).  
  - Text encoder: Hugging Face (e.g. `bert-base-uncased`) loaded at runtime.

---

## 7. Summary

| Aspect | Description |
|--------|-------------|
| **Architecture** | Pipeline: annotation + video → frame extraction (OpenCV) → per-frame GroundingDINO inference → comparison with ground-truth ranges → precision/recall/accuracy. |
| **Data** | Single annotation file with video name, label, and inclusive frame ranges; per-frame boolean ground truth derived from these ranges. |
| **ML** | Pre-trained open-vocabulary detector (GroundingDINO, Swin-T + BERT); text prompt “smoke or fire”; frame-level binary anomaly decision via detection presence and thresholds. |
| **Flow** | `video_frame_extractor` for data and viz; `GroundingDINO_Video_Frame_Analysis` for full detect-and-evaluate pipeline; `GroundingDINO` for detector-only demos. |

This technical document reflects the application architecture, implementation flow, and machine learning concepts utilized in the Computer Vision Lab project as implemented in the current codebase.
