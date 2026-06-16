# Physio-Deepfake

Official repository for deepfake detection using physiological features. This research demonstrates that biological signals (rPPG, Blink Rate, Breathing Rate, Micro-movements) are more robust indicators of deepfake content on small datasets than traditional visual features or standard recurrent neural networks.

## Research Results

The following table summarizes the performance of various approaches on the SDFVD dataset (106 videos). **All results were obtained using 10-fold GroupKFold validation** to ensure no data leakage between real and fake pairs of the same video and high statistical power.

| Rank | Approach | Features / Architecture | Accuracy (± SD) | 95% CI (Accuracy) | Mean AUC | Statistical Significance |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **EfficientNet + CRNN** | Frozen Backbone + Recurrent + Attn | **76.75% ± 14.88%** | [66.10%, 87.40%] | **0.8230** | **Significant (p=0.0003)** |
| 2 | **Physiological ML (Ours)** | **rPPG, Blink, Breath, Micro-mvmnts** | **73.00% ± 8.20%** | [67.14%, 78.86%] | **0.7145** | **Significant (p=0.002)** |
| 3 | **PCGN (Graph + Base Combined)** | PCGN Edges + 47 Autonomic Features | **68.25% ± 8.88%** | [61.90%, 74.60%] | **0.6943** | **Significant (p=0.0002)** |
| 4 | **PCGN (Graph Only)** | 6 Autonomic Synchronization Edges | **66.25% ± 8.75%** | [59.99%, 72.51%] | **0.5373** | **Significant (p=0.0003)** |
| 5 | XceptionNet | Fine-tuned ImageNet Pretrained CNN | 57.78% ± 13.66% | [48.01%, 67.55%] | 0.6653 | Not Significant (p=0.105) |
| 6 | Physiological (Ensemble v1) | Joint Biological Signal Probability | 56.80% ± 9.50% | [50.00%, 63.60%] | 0.5520 | Not Significant |
| 7 | Traditional Visual ML | Landmarks + Color Histograms | 53.33% ± 11.72% | [44.49%, 62.17%] | 0.4654 | Not Significant (p=0.416) |
| 8 | Stacked CRNN | MobileNetV2 Backbone + BiLSTM + Attn | 51.67% ± 2.68% | [49.75%, 53.59%] | 0.5000 | Not Significant (p=0.081) |
| 9 | ResNet50 | Fine-tuned ImageNet Pretrained CNN | 50.91% ± 1.82% | [48.39%, 53.43%] | 0.5370 | Not Significant (p=0.374) |

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Nix package manager (optional, for developer shell environment)
- [MediaPipe Face Landmarker Task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) (included in parent directory)

### 2. Installation
To load the project dependencies under Nix develop shell (highly recommended for reproducibility):
```bash
nix develop
```

Or install dependencies manually via `pip`:
```bash
pip install -r requirements.txt
```

### 3. Usage

#### Run Physiological Detection (Our Best Hand-Crafted Model: 73.0%)
```bash
python src/physio_detection.py --dataset path/to/dataset
```

#### Run EfficientNet + CRNN Baseline (Best DL Model: 76.75%)
```bash
python src/efficientnet_crnn.py --dataset path/to/dataset
```

#### Run Physiological Consistency Graph Network (PCGN) Pipeline
```bash
python scripts/run_graph_consistency.py
```

#### Run Learning Curve Analysis
```bash
python scripts/run_learning_curves.py
```

#### Run Feature Ablation Study
```bash
python scripts/run_ablation_study.py
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
- `src/`: Core implementation of the physiological detection model, neural networks, and baselines.
- `scripts/`: Scripts for PCGN graph modeling, learning curves, and feature ablation studies.
- `results/`: Verifiable JSON outputs and cached features from all experimental runs.
- `visualizations/`: Plots showing learning curves, SHAP attribution, and model comparison.
- `face_landmarker.task`: Required model for facial landmark extraction (resolves to parent directory).
- `flake.nix` & `flake.lock`: Nix configurations for developer environment reproducibility.

---

## Methodology
Unlike previous approaches that suffered from train-test leakage, this research employs **GroupKFold** cross-validation based on video IDs. This ensures that a model never sees a version (real or fake) of a video in the training set that it will be tested on in the validation set.

This rigorous approach reveals that:
1. **Standard Neural Networks (CRNN, ResNet, Xception)** struggle to generalize on small datasets (< 150 samples), essentially performing near random chance.
2. **Specialized Frozen Backbones (EfficientNet + CRNN)** can achieve high accuracy (76.75%), but exhibit extreme sensitivity to configuration details and preprocessing.
3. **Physiological Signals (Ours)** provide a highly stable, interpretable, and data-efficient biometric representation, achieving strong statistical significance with extremely limited data.
4. **Physiological Consistency Graph Networks (PCGN)** successfully evaluate coupling and synchronization between different autonomic subsystems (heartbeat, eye blinks, breathing, and head movements), proving to be a highly viable future direction.

---

## Data Access
The SDFVD (Small-scale Deepfake Forgery Video Dataset) used in this research must be obtained separately due to licensing.
1. Download the dataset from the original repository or host.
2. Extract the videos into `sdfvd_dataset/SDFVD/videos_real` and `sdfvd_dataset/SDFVD/videos_fake`.
3. Ensure video files follow the naming convention `v{id}.mp4` (real) and `vs{id}.mp4` (fake).
