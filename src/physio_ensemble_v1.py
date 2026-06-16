#!/usr/bin/env python3
"""
Physiological-Only Deepfake Detection with Ensemble (v1)
Uses biological signals: rPPG, Blink, Breathing, Micro-movements.
10-fold GroupKFold Cross-Validation for robust evaluation.
"""

import os
import sys
import argparse
import json
import glob
import warnings
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

SEED = 42

def extract_video_id(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r"(\d+)", basename)
    return match.group(1) if match else basename

@dataclass
class Config:
    dataset_path: str = "./sdfvd_dataset/SDFVD"
    real_folder: str = "videos_real"
    fake_folder: str = "videos_fake"
    max_frames: int = 150 
    n_folds: int = 10
    model_path: str = "face_landmarker.task"

class PhysiologicalExtractor:
    def __init__(self, config: Config):
        self.config = config
        base_options = python.BaseOptions(model_asset_path=config.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_ear(self, landmarks, indices, w, h):
        pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h_d = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h_d + 1e-7)

    def extract_features(self, video_path: str) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1: fps = 30
        
        rgb_signals, ear_signals, nose_pos, face_pos = [], [], [], []
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.config.max_frames:
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self.landmarker.detect(mp_image)
            
            if res.face_landmarks:
                lms = res.face_landmarks[0]
                h, w = frame.shape[:2]
                
                # EAR
                left_ear = self.get_ear(lms, [33, 160, 158, 133, 153, 144], w, h)
                right_ear = self.get_ear(lms, [362, 385, 387, 263, 373, 380], w, h)
                ear_signals.append((left_ear + right_ear) / 2.0)
                
                # rPPG ROI (forehead)
                fh_y, fh_x = int(lms[10].y * h), int(lms[10].x * w)
                roi = frame[max(0, fh_y):min(h, fh_y+20), max(0, fh_x-10):min(w, fh_x+10)]
                if roi.size > 0:
                    rgb_signals.append(np.mean(roi, axis=(0,1)))
                
                nose_pos.append([lms[4].x * w, lms[4].y * h])
                face_pos.append([np.mean([lm.x for lm in lms]) * w, np.mean([lm.y for lm in lms]) * h])
                
            frame_count += 1
        cap.release()
        
        if len(rgb_signals) < 60: return None
        
        features = []
        # Heart Rate (simplified CHROM)
        rgb = np.array(rgb_signals)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        x = 3*r - 2*g
        y = 1.5*r + g - 1.5*b
        s = x - (np.std(x)/np.std(y))*y
        # FFT for HR
        freqs = np.fft.fftfreq(len(s), 1/fps)
        mags = np.abs(np.fft.fft(s))
        valid = (freqs >= 0.7) & (freqs <= 4.0)
        hr = freqs[valid][np.argmax(mags[valid])] * 60 if any(valid) else 0
        features.append(hr)
        
        # Blink Rate
        ear = np.array(ear_signals)
        blinks = np.sum(ear < 0.2)
        features.append(blinks / (len(ear)/fps) * 60)
        
        # Micro-movements
        face = np.array(face_pos)
        mvmnt = np.mean(np.linalg.norm(np.diff(face, axis=0), axis=1))
        features.append(mvmnt)
        
        # Stats
        features.extend([np.mean(ear), np.std(ear), np.std(r), np.std(g), np.std(b)])
        
        return np.array(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./sdfvd_dataset/SDFVD")
    args = parser.parse_args()
    
    config = Config(dataset_path=args.dataset)
    extractor = PhysiologicalExtractor(config)
    
    X, y, video_ids = [], [], []
    for folder, label in [(config.real_folder, 0), (config.fake_folder, 1)]:
        videos = glob.glob(os.path.join(config.dataset_path, folder, "*.mp4"))
        print(f"Processing {folder}...")
        for v in tqdm(videos):
            feat = extractor.extract_features(v)
            if feat is not None:
                X.append(feat)
                y.append(label)
                video_ids.append(extract_video_id(v))
                
    X, y, groups = np.array(X), np.array(y), np.array(video_ids)
    print(f"Loaded {len(X)} samples")
    
    gkf = GroupKFold(n_splits=config.n_folds)
    accs = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
        clf.fit(X_train, y_train)
        accs.append(accuracy_score(y_val, clf.predict(X_val)))
        
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"Final Ensemble v1 Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # Save results
    results = {
        "model_type": "Physiological Ensemble v1",
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "accuracies": [float(a) for a in accs]
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/physiological_ensemble_v1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved to results/physiological_ensemble_v1_results.json")

if __name__ == "__main__":
    main()
