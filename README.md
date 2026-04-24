# Physio-Deepfake

Official repository for deepfake detection using physiological features. This research demonstrates that biological signals (rPPG, Blink Rate, Breathing Rate, Micro-movements) are more robust indicators of deepfake content on small datasets than traditional visual features or neural networks.

## Research Results

The following table summarizes the performance of various approaches on the SDFVD dataset (106 videos). **All results were obtained using 10-fold GroupKFold validation** to ensure no data leakage between real and fake pairs of the same video and high statistical power.

| Rank | Approach | Features | Accuracy | Statistical Significance (p < 0.05) |
| :--- | :--- | :--- | :---: | :---: |
| 1 | **Physiological (Final)** | **rPPG, Blink, Breath, Micro-mvmnts** | **73.0%** | **Significant (p=0.002)** |
| 2 | Physiological (Ensemble) | Standard biological signals | 56.8% | Marginal |
| 3 | Visual ML (Proper) | Landmarks + Color Histograms | 53.3% | Not Significant |
| 4 | XceptionNet (Baseline) | Standard CNN Fine-tuned | 52.0% | Not Significant |
| 5 | CRNN (Stacked) | MobileNetV2 + BiLSTM + Attention | 51.7% | Not Significant |

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- [MediaPipe Face Landmarker Task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) (included in repo)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Usage

#### Run Physiological Detection (Best Model: 73.0%)
```bash
python src/physio_detection.py --dataset path/to/dataset
```

#### Run Physiological Ensemble v1 (56.8%)
```bash
python src/physio_ensemble_v1.py --dataset path/to/dataset
```

#### Run Visual Baseline (53.3%)
```bash
python src/visual_baseline.py --dataset path/to/dataset
```

#### Run Stacked CRNN (51.7%)
```bash
python src/crnn_stacked.py --dataset path/to/dataset
```

---

## Repository Structure
- `src/`: Core implementation of the physiological detection model and visual baseline.
- `results/`: Verifiable JSON outputs from all experimental runs.
- `docs/`: Detailed research summary and methodology.
- `face_landmarker.task`: Required model for facial landmark extraction.
- `devenv.nix` & `pyproject.toml`: Environment configuration for reproducible research.

---

## Methodology
Unlike previous approaches that suffered from data leakage, this research employs **GroupKFold** cross-validation based on video IDs. This ensures that a model never sees a version (real or fake) of a video in the training set that it will be tested on in the validation set. 

This rigorous approach reveals that:
1. **Neural Networks (CRNN, Xception)** struggle to generalize on small datasets (< 500 samples), essentially performing at random chance.
2. **Traditional Visual Features** (Landmarks/Texture) are easily bypassed by modern deepfake generation methods.
3. **Physiological Signals** provide a more stable and discriminative feature set for forensic analysis, achieving significance even with limited data.

---

## Data Access
The SDFVD (Small-scale Deepfake Forgery Video Dataset) used in this research must be obtained separately due to licensing. 
1. Download the dataset from [Original Source Link].
2. Extract the videos into `sdfvd_dataset/SDFVD/videos_real` and `sdfvd_dataset/SDFVD/videos_fake`.
3. Ensure video files follow the naming convention `v{id}.mp4` (real) and `vs{id}.mp4` (fake).



