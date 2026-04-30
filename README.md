# BeeVision

**A Morphology-Guided Two-Stage Pipeline for Real-Time Bilateral Antenna Keypoint Tracking and Behavioral Analysis in Free-Moving Honeybees**

<p align="center">
  <img src="gifs/01_single_bee.gif" width="45%" alt="Single bee tracking" />
  <img src="gifs/02_multibee.gif" width="45%" alt="Multi-bee tracking" />
</p>

BeeVision is the first system to achieve real-time bilateral antenna keypoint tracking in free-moving, unmarked honeybees within living observation hives. By combining a custom-trained YOLO11n-pose model for body keypoints with a dedicated morphological refinement stage for antenna localization, BeeVision closes the 89 percentage-point train-deploy gap that has structurally prevented sub-pixel-width antenna tracking in standard heatmap-based pose estimation frameworks.

---

## Table of Contents

- [Highlights](#highlights)
- [Why BeeVision](#why-beevision)
- [Repository Contents](#repository-contents)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Downloads](#downloads)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Architecture](#pipeline-architecture)
- [Behavioral Analysis](#behavioral-analysis)
- [Training Your Own Model](#training-your-own-model)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Parameter Reference](#parameter-reference)
- [Hardware Requirements](#hardware-requirements)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Funding](#funding)
- [License](#license)
- [Contact](#contact)

---

## Highlights

| Metric | YOLO-only baseline | BeeVision (two-stage) | Gain |
|---|---|---|---|
| Antenna PCK@10 | 10.7% | **92.7%** | +82.0 pp |
| Antenna RMSE | 29.3 px | **6.7 px** | 4.4× lower |
| Antenna F1 (worst keypoint) | 55.4% | **99.3%** | +43.9 pp |
| Errors > 25 px | 49.7% | **2.3%** | −47.4 pp |
| Bilateral tracking rate | — | **98.9%** | — |
| Real-time throughput | — | **≥15 fps up to 14 bees** | — |

Evaluated on 2,646 instances spanning workers, queens, and drones. Long-duration stability validated over 60 continuous minutes with 52 unique bees and zero statistical drift in any accuracy or resource metric.

---

## Why BeeVision

Honeybee colony mortality reached 55.6% in the 2024–2025 season, with colony collapse disorder (CCD) surging 110% and now accounting for 55.5% of all winter losses. Crucially, CCD is currently diagnosed only post-mortem — no instrument exists to detect the sub-lethal cognitive changes that precede colony collapse in individual bees within a natural colony.

The behavioral signal that makes individual-level monitoring possible lies in the antennae. Honeybees exhibit pronounced **olfactory lateralization**: the right antenna dominates short-term learning and recall, while the left antenna becomes dominant for long-term memory consolidation. Dissociation between the two antennae constitutes a sensitive early indicator of cognitive impairment caused by pesticide exposure or viral infection — but capturing this asymmetry has previously required wax immobilization, antenna painting, or surgical blocking, all of which confound the very cognitive state being measured.

BeeVision removes that constraint entirely by enabling continuous, markerless, bilateral antenna tracking across all individuals in a living colony simultaneously.

---

## Repository Contents

```
BeeVision/
│
├── README.md                          ← you are here
├── LICENSE                            ← MIT
│
├── beevision/
│   └── bee_pose_test_result_best.py   ← the full pipeline (PyQt6 GUI + inference)
│
├── weights/                           ← place trained .pt model here (download below)
├── dataset/                           ← training + evaluation data (download below)
│   ├── training/                      ← 3,535 instances for retraining
│   └── evaluation/                    ← 2,646 instances for paper Tables 1–3
│
├── test_videos/                       ← short demo clips (download below)
│
├── results/                           ← qualitative outputs from the paper figures
│   ├── Single Bee/                    ← Fig. 12 A–F
│   ├── Single Bee Qualitative/
│   ├── Multibee/                      ← Fig. 12 G–J
│   ├── LR RL/                         ← Fig. 12 K–N (cross-pattern contact)
│   ├── RR/                            ← Fig. 12 O–R (right–right contact)
│   ├── LL/                            ← Fig. 12 S–V (left–left contact)
│   ├── Trophallaxis/                  ← Fig. 12 W–Z (food transfer event)
│   ├── Head Contact/                  ← Fig. 13 A–E (Tier 3 anterior region)
│   ├── Thorax Abdomen Contact/        ← Fig. 13 F–J (Tier 3 posterior regions)
│   ├── Drone Qualitative/             ← Fig. 13 L (cross-caste, drone)
│   └── Full frame Qualitative/        ← Fig. 13 N (crowded colony scene)
│
└── gifs/                              ← animated demonstrations of each capability
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/BeeVision.git
cd BeeVision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights, dataset, and test videos (see Downloads section)

# 4. Run the GUI
python beevision/bee_pose_test_result_best.py
```

In the GUI:
1. Click **Select Model** → choose `weights/beevision_yolo11n_pose.pt`
2. Click **Select Video Source** → choose any clip from `test_videos/`
3. Click **Run**

Switch between the tabs (Pose Estimation, Darkest Pixels, ROI + BBox + Body, For Research, etc.) to see every stage of the pipeline in real time.

---

## Installation

### Requirements

- **Python** 3.10 or higher
- **CUDA** 11.8+ (optional, for GPU acceleration — strongly recommended)
- **OS** Linux (tested on Ubuntu 22.04), Windows 10/11, macOS 12+

### Core dependencies

```bash
pip install -r requirements.txt
```

The core requirements include:

```
torch>=2.0
ultralytics>=8.0.196
opencv-contrib-python>=4.8.0    # contrib build is required for cv2.ximgproc
PyQt6>=6.5
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

> **Important:** Use `opencv-contrib-python`, not the base `opencv-python`. BeeVision uses `cv2.ximgproc.thinning` for fast Zhang-Suen skeletonization, which is only available in the contrib build. The script falls back to a slower vectorized NumPy implementation if `ximgproc` is unavailable, but the contrib build is recommended for real-time performance.

### Optional acceleration

For maximum throughput on CUDA-equipped systems, install the optional acceleration dependencies:

```bash
pip install -r requirements-optional.txt
```

Optional dependencies:

```
cupy-cuda12x>=12.0    # GPU-accelerated NumPy operations
numba>=0.58           # JIT compilation for triangle-membership tests
```

The pipeline auto-detects these at import time and falls back gracefully to CPU/NumPy when they're absent.

### Verify your installation

```bash
python -c "import torch, cv2, ultralytics, PyQt6; \
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); \
print('OpenCV:', cv2.__version__, '| ximgproc:', hasattr(cv2, 'ximgproc')); \
print('Ultralytics:', ultralytics.__version__)"
```

Expected output on a working install:
```
PyTorch: 2.1.0+cu118 | CUDA: True
OpenCV: 4.8.1 | ximgproc: True
Ultralytics: 8.0.196
```

---

## Downloads

All large assets (model weights, dataset, test videos) are hosted on Google Drive due to GitHub's file size limits.

### 1. Trained Model Weights

YOLO11n-pose weights trained on 3,535 manually annotated honeybee instances across workers, queens, and drones.

**Download:** https://drive.google.com/drive/folders/1Jwxrdot9pSg63l6OPYQKcUpMNiBeGo_m?usp=sharing

After downloading, place the `.pt` file in `weights/` so the path resolves to:

```
weights/beevision_yolo11n_pose.pt
```

### 2. Training Dataset

3,535 manually annotated instances in COCO format, used to train the YOLO11n-pose backbone. Splits: workers and drones partitioned 645 / 177 / 91 (train / valid / test); queens partitioned 305 / 87 / 44.

**Download:** https://drive.google.com/drive/folders/1yyn6zrlAe2xXj5S0ceutEPtbt06XrBDD?usp=sharing

Extract into `dataset/training/`:

```
dataset/training/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

### 3. Evaluation Dataset

The held-out evaluation set used to compute Tables 1, 2, and 3 in the paper. 2,646 instances spanning workers (1,302), drones (1,140), and queens (204), yielding 10,584 antenna keypoint annotations.

**Download:** https://drive.google.com/drive/folders/1cyQv3DHgzcmMGbtvgJodotjqgOLdqwUp?usp=sharing

Extract into `dataset/evaluation/`.

### 4. Test Videos

Short, unannotated demonstration clips for runnable validation of the pipeline.

**Download:** https://drive.google.com/drive/folders/1gc34X0bcShafH4pRDbm8Y2gpA5jVM7nP?usp=sharing

Extract into `test_videos/`. These clips are recorded under the same conditions as the training dataset (Phantom VRI-MIRO-C321-16GB-M at 500 fps, 94 cm working distance, downsampled to 640×480).

> **Note:** Test videos are **unannotated** demonstration clips intended for end-to-end pipeline validation only. To reproduce paper metrics (Tables 1–3), use the held-out evaluation dataset (download #3) via the GUI's Evaluation tab.

---

## Running the Pipeline

### GUI Mode (default)

```bash
python beevision/bee_pose_test_result_best.py
```

The GUI exposes nine visualization tabs:

| Tab | Purpose |
|---|---|
| **Pose Estimation** | Final tracking output with all 9 keypoints, bounding boxes, and identity labels |
| **Darkest Pixels & ROI** | Antenna candidate detection within ROI polygons |
| **Darkest Mask (Viridis)** | Heatmap visualization of the morphological darkness response |
| **Darkest Map (No Keypoints)** | Raw morphological output, useful for parameter tuning |
| **Body Keypoints + ROI Vectors** | Body keypoint stability check with ROI direction vectors overlaid |
| **ROI + BBox + Body** | ROI polygons, bounding boxes, and body keypoints together |
| **Grayscale + Body + ROI** | Inspection view for verifying ROI angle stability |
| **Full Frame Darkest** | Whole-frame morphological response without per-bee cropping |
| **🔬 For Research** | The 13-panel grid showing every pipeline stage A–M (Figs. 5 and 6 of the paper) |

The right-hand sidebar contains:
- **Configuration** — model selector, video selector, threshold sliders, run/stop controls
- **Analysis** — live trophallaxis events, antenna contact statistics, regional contact data
- **Evaluation** — load a COCO annotation file to evaluate predictions against ground truth

### Sliders

| Slider | Default | Range | Effect |
|---|---|---|---|
| Kernel Size | 5 | 3–21 px | Tophat structuring element size; larger = thicker antennae |
| Smoothing Frames | 3 | 1–10 | Body keypoint EMA window (Eq. 1); higher = smoother but more lag |
| ROI Triangle Thickness | 1 | 0–5 | Visualization only; 0 hides the polygons |
| ROI Percentile | 0 (Otsu) | 0–50% | Per-ROI percentile threshold; 15% matches paper default |
| Frame Darkness | 0 | 0–100% | Pre-processing dark-pixel attenuation, applied before YOLO |

> **Paper-default kernel size is 15.** The GUI default of 5 is tuned for the thinnest worker antennae in close-up imaging. For paper-faithful reproduction, set kernel size to 15 and ROI percentile to 15.

### Image Folder Mode

In addition to video files, BeeVision can process a folder of still images as a virtual 60 fps stream:

1. Click **Select Video Source**
2. Choose **"Folder of images"** in the dialog
3. Pick a directory containing `.jpg`, `.png`, `.bmp`, `.tiff`, or `.webp` files

Images are read in lexicographic filename order. This is how the paper's 500 fps Phantom recordings are processed.

---

## Pipeline Architecture

BeeVision is a three-stage cascade running on every frame independently per detected bee.

<p align="center">
  <img src="gifs/03_full_frame.gif" width="80%" alt="Full pipeline running on crowded colony scene" />
</p>

### Stage 1 — YOLO11n-pose detection

A custom-trained CSPDarkNet + FPN backbone (YOLO11n-pose) detects nine anatomical keypoints per bee:

```
k0  head (frons)
k1  prothorax
k2  mesothorax
k3  metathorax
k4  terminal abdomen
k5  right antenna joint (scape-pedicel)
k6  right antenna tip (flagellum)
k7  left antenna joint
k8  left antenna tip
```

Identity persistence is maintained by **ByteTrack** with Hungarian assignment and Kalman state estimation. Body keypoints (k0–k4) are stabilized by a 3-frame confidence-weighted moving average; antenna keypoints (k5–k8) are excluded because Stage 2 overwrites them every frame.

### Stage 2 — Morphological antenna refinement

Stage 1's antenna predictions are discarded and replaced with geometrically grounded estimates derived directly from image gradients:

1. **ROI construction.** Two polygonal ROIs (left and right) are anchored at k0, with rays cast at ±90° from the smoothed body heading and clipped to a 1.25×-expanded bounding box.
2. **Multi-directional tophat filtering.** Three directional tophats (vertical 15×1, horizontal 1×15, diagonal 9×9 cross) applied to the inverted grayscale frame, combined by element-wise maximum.
3. **Per-ROI percentile thresholding.** The top 15% brightest pixels within each ROI polygon survive, followed by strongest-component selection scored by `mean_intensity × √area`.
4. **Dual-antenna selection.** The two highest-scoring connected components in the head region are selected and assigned to left/right via perpendicular projection on the body's lateral axis.
5. **Skeletonization.** Zhang-Suen thinning (via `cv2.ximgproc`) with dilate-and-re-skeletonize gap bridging produces a 1-pixel-wide medial axis.
6. **BFS path tracing.** Breadth-first search from the head-anchored base to the optimal endpoint, where the endpoint is selected by a blended score: `0.5 × path_length + 0.3 × tangent_alignment + 0.2 × heading_projection`.
7. **Keypoint placement.** Joint at 38% of the path length (corresponding to the scape-pedicel articulation, confirmed by SEM); tip at 100%.

### Stage 3 — Optical flow tracking layer

A pyramidal Lucas-Kanade tracker (window 15×15, 3-level pyramid, 10 iterations per frame) runs in parallel with Stage 2, providing temporal continuity during transient occlusion. Three biological plausibility checks gate every update:

- **V1 — Anatomical constraints.** Joints 10–60 px from head, tips 40–120 px, joint-before-tip ordering enforced.
- **V2 — Morphological darkness.** Each tracked point must lie on a dark image structure (`255 − I ≥ 30`).
- **V3 — Motion limits.** Frame-to-frame displacement bounded at 50 px (joints) / 80 px (tips).

Confidence is updated asymmetrically: `+0.02` per pass, `−0.30` per validation failure, `−0.40` per catastrophic flow failure. Re-initialization from Stage 2 triggers when confidence falls below 0.4 or every 120 frames, whichever comes first.

---

## Behavioral Analysis

Beyond keypoint tracking, BeeVision extracts a comprehensive behavioral metrics suite directly from the geometric configuration of tracked keypoints — no learned classifier is required, and every event is visually verifiable.

### Directional contact patterns

<p align="center">
  <img src="gifs/04_RR_contact.gif" width="32%" alt="RR contact" />
  <img src="gifs/05_LL_contact.gif" width="32%" alt="LL contact" />
  <img src="gifs/06_LR_RL_cross.gif" width="32%" alt="Cross-pattern contact" />
</p>

For every bee pair (A, B), antenna contacts are classified into four directional patterns based on which antenna of each bee is involved:

| Pattern | Behavioral correlate |
|---|---|
| **RR** (right–right) | Affiliative behaviors: trophallaxis, allogrooming, food sharing |
| **LL** (left–left) | Threat awareness; high LL ratios indicate over-extended defensive states |
| **RL / LR** (cross) | Stress, defensive posturing, aggressive encounters |

The **lateralization index** `LI = (RR − LL) / Total` quantifies asymmetry on `[−1, +1]`. The **cross-pattern stress ratio** `ρ_cross = (RL + LR) / Total` captures defensive arousal — controlled perturbation experiments produced `ρ_cross = 0.43 ± 0.08` versus baseline `0.16 ± 0.05` (p < 0.0001).

### Regional contact detection

<p align="center">
  <img src="gifs/07_head_contact.gif" width="48%" alt="Antenna-to-head contact" />
  <img src="gifs/08_thorax_abdomen_contact.gif" width="48%" alt="Antenna-to-posterior contact" />
</p>

Each bee's body is partitioned into four anatomical segments (prothorax, mesothorax, metathorax, abdomen). Antenna tips landing within 10 px of any segment trigger a regional contact event. Elevated prothorax contacts suggest head-focused inspection or aggression assessment; elevated abdominal contacts indicate posterior-focused interactions such as gland secretion transfer.

### Trophallaxis event detection

<p align="center">
  <img src="gifs/09_trophallaxis.gif" width="80%" alt="Trophallaxis event detection" />
</p>

Trophallaxis (mouth-to-mouth food transfer) events are confirmed when **all five conditions** are simultaneously satisfied for at least 10 seconds:

1. Both bees in extended posture (`straightness ≥ 0.70`)
2. Antiparallel body alignment (`120° ≤ Δθ ≤ 240°`)
3. Sustained duration (`T ≥ 10 s`)
4. Active antenna contact present
5. Head proximity within `[20, 80]` pixels

Confirmed events receive a quality score combining straightness, duration, and right-antenna dominance. Observed scores span 42–147; `Q > 120` identifies high-quality food-transfer events (top 18.7% of verified interactions).

### Cross-caste tracking

<p align="center">
  <img src="gifs/10_drone_queen.gif" width="80%" alt="Drone and queen tracking" />
</p>

Identical pipeline parameters track all three colony castes. Performance on the evaluation set:

| Caste | Orientation | n | Antenna RMSE | PCK@10 |
|---|---|---|---|---|
| Worker | Dorsal | 6,350 | 6.2 px | 93.7% |
| Queen | Dorsal | 816 | 7.1 px | 90.1% |
| Drone | Dorsal | 5,345 | 14.1 px | 70.7% |

The drone gap reflects training data volume (202 instances vs. 2,897 worker instances), not morphological dissimilarity — additional drone annotations close the gap directly.

---

## Training Your Own Model

To retrain YOLO11n-pose on the BeeVision dataset:

```bash
yolo pose train \
  model=yolo11n-pose.pt \
  data=dataset/training/data.yaml \
  imgsz=640 \
  batch=16 \
  epochs=500 \
  patience=50 \
  optimizer=AdamW \
  lr0=0.001 \
  weight_decay=0.0005 \
  cos_lr=True \
  mosaic=1.0 \
  mixup=0.1 \
  copy_paste=0.3 \
  hsv_h=0.04 \
  hsv_s=0.5 \
  hsv_v=0.3 \
  project=runs/train \
  name=beevision
```

Hyperparameters match the paper (Section 2.2.3). Training in the paper converged at epoch 401 (validation plateau, early stopping). Final pose mAP@0.5 = 0.988, box mAP@0.5 = 0.986. Best weights are written to `runs/train/beevision/weights/best.pt`.

The dataset uses the 9-keypoint schema with `flip_idx = [0, 3, 4, 1, 2, 7, 8, 5, 6]` so that horizontal-flip augmentation correctly swaps left/right anatomical pairs (k1↔k3, k5↔k7, k6↔k8).

---

## Reproducing Paper Results

To reproduce the headline numbers in Tables 1, 2, and 3:

1. Download the **Evaluation Dataset** (link above) into `dataset/evaluation/`.
2. Download the trained **Model Weights** into `weights/`.
3. Launch the GUI: `python beevision/bee_pose_test_result_best.py`.
4. Switch to the **Evaluation** tab in the right sidebar.
5. Click **Select Model** → `weights/beevision_yolo11n_pose.pt`.
6. Click **Select COCO Annotations** → the `_annotations.coco.json` from the evaluation set.
7. Click **Select Image Folder** → the matching `images/` directory.
8. Click **Run** for the YOLO-only baseline, then **Run Full Pipeline** for the two-stage system.

The Evaluation tab will produce side-by-side PCK curves, per-keypoint RMSE bars, error severity distributions, and downloadable CSVs matching the paper's reported metrics.

Expected results on the worker subset (1,302 instances):

| Configuration | Antenna PCK@10 | Antenna RMSE |
|---|---|---|
| YOLO-only | 10.7% | 29.3 px |
| BeeVision (two-stage) | 92.7% | 6.7 px |

---

## Parameter Reference

The pipeline exposes ~30 parameters across morphological preprocessing, optical flow validation, behavioral thresholds, and visualization. The most important ones, with their paper-default values:

### Morphological preprocessing

| Parameter | Paper default | Range | Notes |
|---|---|---|---|
| Tophat kernel size | 15 px | 3–21 | Robust plateau 11–19 px (Table S1) |
| ROI percentile (`p`) | 15% | 12–18% | Tightest tunable parameter |
| Min antenna area (`A_min`) | 8 px | 5–15 | Below this = noise |
| Min elongation (`ε_min`) | 1.2 | 1.0–2.0 | Flat response across this range |

### Optical flow validation

| Parameter | Paper default | Notes |
|---|---|---|
| LK window size | 15×15 px | Pyramidal, 3 levels |
| Iterations per frame | 10 | ε = 0.03 px |
| Anatomical bounds (joint) | 10–60 px from head | V1 check |
| Anatomical bounds (tip) | 40–120 px from head | V1 check |
| Darkness threshold | `255 − I ≥ 30` | V2 check |
| Motion bounds | 50 px (joint) / 80 px (tip) | V3 check |
| Confidence increment | +0.02 / pass | Asymmetric — pessimistic |
| Confidence decrement | −0.30 / fail | |
| Catastrophic decrement | −0.40 / no-output | Triggers re-init |
| Re-initialization threshold | C < 0.4 | Or every 120 frames |

### Trophallaxis detection

| Parameter | Paper default |
|---|---|
| Min straightness | 0.70 |
| Angular alignment range | 120°–240° |
| Min duration | 10.0 seconds |
| Antenna contact threshold | 10 px |
| Head proximity range | 20–80 px |

### Behavioral thresholds

| Parameter | Paper default |
|---|---|
| Regional contact threshold | 10 px |
| Behavioral analysis window | 300 frames (10 s @ 30 fps) |
| ID switch grace period | 30 frames |
| Cross-pattern stress trigger | `ρ_cross > 0.35` |

> **Implementation note:** Some parameters in the released code are tuned slightly differently from the paper-stated values for deployment robustness (e.g. widened optical flow anatomical bounds, gentler confidence decay paired with a 12-frame grace period). For paper-faithful reproduction, set the kernel size to 15 and ROI percentile to 15% in the GUI sliders.

---

## Hardware Requirements

BeeVision was developed and benchmarked on a **mid-range consumer laptop** to ensure deployability without specialized infrastructure.

| Component | Tested | Minimum |
|---|---|---|
| CPU | AMD Ryzen 7 4800H (8 cores, 2.9 GHz) | 4 cores @ 2.5 GHz |
| RAM | 16 GB | 8 GB |
| GPU | NVIDIA RTX 3050 Laptop (4 GB VRAM, 60 W TDP) | Any CUDA 11+ GPU with ≥2 GB VRAM (or CPU-only) |
| CUDA | 12.4 | 11.8+ |
| Disk | 5 GB free | 5 GB free |

Performance characteristics on the tested hardware:

- **Real-time boundary:** 14 simultaneously tracked bees at ≥15 fps
- **Linear scaling:** `T(N) = 19.3 + 3.24 × N` ms (R² = 0.9994)
- **Stage 2 = 56.1% of pipeline time** (CPU-bound — GPU upgrades alone do not improve throughput; CPU parallelization is the primary scaling avenue)
- **Memory:** 2.08 GB RAM, 847 MB GPU VRAM, both flat over 60-minute sessions

---

## Limitations

These define the boundary of the current contribution and the roadmap for future work:

1. **Ventral-view performance.** Workers and drones imaged from the ventral side exhibit a consistent 30–40 pp PCK@10 drop because tophat preprocessing parameters were optimized for dorsal contrast geometry. Orientation-aware parameter switching (triggered by the YOLO body axis estimate) is the priority corrective.

2. **Drone localization.** Drone PCK@10 of 70.7% lags worker performance proportional to training data volume (202 vs. 2,897 instances), not morphological dissimilarity. Targeted annotation expansion closes this gap directly.

3. **Temporal resolution.** The 15 fps pipeline output rate places the Nyquist limit at 7.5 Hz, capturing the lower portion of the 5–15 Hz antenna scanning range. CPU parallelization of Stage 2 is the primary throughput avenue.

4. **Behavioral metric validation.** The suite of lateralization, regional contact, and trophallaxis metrics requires large-scale ground-truth annotation campaigns to establish population-level norms before clinical deployment as a colony health indicator.

---

## Troubleshooting

### `cv2.ximgproc` not available

You installed `opencv-python` instead of `opencv-contrib-python`. Fix:

```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python>=4.8.0
```

### Qt font crash on Ubuntu / `OpenType font format error`

The script proactively suppresses this — set the matplotlib backend to Agg and clear the matplotlib font cache:

```bash
rm -rf ~/.cache/matplotlib/fontlist*.json
```

### CUDA out of memory during evaluation

Lower the `max_bees_per_frame` parameter or process in batches. The default of 50 simultaneous detections requires roughly 850 MB of VRAM.

### Model loads but produces no detections

Confirm the input video matches the training conditions: monochrome or grayscale-equivalent, top-down hive view, bees visible at roughly 60–120 px body length. Side-view or ventral-view footage will produce reduced detection rates.

### GUI launches but freezes immediately

Some Linux distributions ship with an Ubuntu font that crashes Qt's font system. The script forces matplotlib to DejaVu Sans at import time; if the freeze persists, run:

```bash
export QT_QPA_PLATFORM=offscreen
python beevision/bee_pose_test_result_best.py
```

This will run the inference pipeline without rendering the GUI and is useful for headless evaluation.

---

## Funding

This material is based upon work supported by the **National Science Foundation** under Award No. **2438295**, and partially supported by the **USDA-Agricultural Research Service**, Project Award No. **59-3060-5-001**.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Shoaib Ahmmad** — `shoaib.ahmmad@ndsu.edu`
PhD Student, Department of Agricultural and Biosystems Engineering
North Dakota State University, Fargo, ND 58108, USA

**Dr. Sulaymon L. Eshkabilov** (corresponding author) — `sulaymon.eshkabilov@ndsu.edu`
Principle Investigator, Agrimechatronics Lab, Department of Agricultural and Biosystems Engineering, NDSU
