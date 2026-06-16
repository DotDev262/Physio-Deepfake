#!/usr/bin/env python3
"""
Physiological Consistency Graph Network (PCGN) - Proof of Concept
Extracts heartbeat, respiration, head motion, and eye dynamics,
computes their synchronization via cross-correlation, and evaluates
the performance of graph features for deepfake detection.
"""

import os
import re
import cv2
import numpy as np
import glob
import random
import json
import warnings
import sys
from scipy import stats, signal
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# Dynamic path resolution relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PARENT_DIR)

from src.physio_detection import EnhancedPhysioFeatureExtractor, find_optimal_threshold

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_PATH = os.path.abspath(os.path.join(PARENT_DIR, "../sdfvd_dataset/SDFVD"))
REAL_FOLDER = "videos_real"
FAKE_FOLDER = "videos_fake"
MODEL_PATH = os.path.abspath(os.path.join(PARENT_DIR, "../face_landmarker.task"))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_video_id(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r"(\d+)", basename)
    return match.group(1) if match else basename

def compute_max_cross_correlation(s1, s2, max_lag=15):
    """Compute maximum absolute cross-correlation over dynamic lags"""
    s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-7)
    s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-7)
    
    best_corr = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = np.mean(s1_norm[-lag:] * s2_norm[:lag])
        elif lag > 0:
            c = np.mean(s1_norm[:-lag] * s2_norm[lag:])
        else:
            c = np.mean(s1_norm * s2_norm)
        if abs(c) > abs(best_corr):
            best_corr = c
    return best_corr

class PCGNFeatureExtractor(EnhancedPhysioFeatureExtractor):
    def __init__(self):
        # We need to construct FaceLandmarker with correct path relative to script parent
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
    def extract_graph_features_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:
            fps = 30
            
        rgb_signals, ear_signals = [], []
        face_pos, nose_pos, lm_seq = [], [], []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.landmarker.detect(mp_image)
            
            if detection_result.face_landmarks:
                lms = detection_result.face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Face position (for head motion)
                face_pos.append(
                    (
                        np.mean([lm.x for lm in lms]) * w,
                        np.mean([lm.y for lm in lms]) * h,
                    )
                )
                
                # Forehead ROI (for heart rate/rPPG)
                fh_y = int(lms[10].y * h)
                fh_x = int(lms[10].x * w)
                roi_h, roi_w = int(0.05 * h), int(0.1 * w)
                fh = frame[max(0, fh_y):min(h, fh_y+roi_h), max(0, fh_x-roi_w//2):min(w, fh_x+roi_w//2)]
                if fh.size > 0:
                    rgb_signals.append(np.mean(fh.reshape(-1, 3), axis=0))
                else:
                    rgb_signals.append([0.0, 0.0, 0.0])

                # EAR (for eye blinking)
                def get_ear(indices):
                    pts = [np.array([lms[i].x * w, lms[i].y * h]) for i in indices]
                    v1 = np.linalg.norm(pts[1] - pts[5])
                    v2 = np.linalg.norm(pts[2] - pts[4])
                    h_d = np.linalg.norm(pts[0] - pts[3])
                    return (v1 + v2) / (2.0 * h_d + 1e-7)

                left_ear = get_ear([33, 160, 158, 133, 153, 144])
                right_ear = get_ear([362, 385, 387, 263, 373, 380])
                ear_signals.append((left_ear + right_ear) / 2.0)
                
                # Nose position (for breathing)
                nose_pos.append((lms[4].x * w, lms[4].y * h))
                lm_seq.append(lms)
            
        cap.release()
        
        # We need a minimum number of frames for meaningful correlation
        if len(rgb_signals) < 60:
            return None
            
        # 1. Base 47 features
        base_features = []
        base_features.extend(self.extract_rppq_features(rgb_signals, fps))
        base_features.extend(self.extract_blink_features(ear_signals, fps))
        base_features.extend(self.extract_breathing_features(nose_pos, fps))
        base_features.extend(self.extract_color_features(rgb_signals))
        base_features.extend(self.extract_micro_movement_features(lm_seq, face_pos))
        base_features = np.array(base_features)
        
        # 2. Extract raw coupled signals
        # Heartbeat: forehead green channel filtered between 0.7-4.0 Hz
        rgb_arr = np.array(rgb_signals)
        g_signal = rgb_arr[:, 1]
        nyquist = 0.5 * fps
        b_c, a_c = signal.butter(4, [0.7 / nyquist, 4.0 / nyquist], btype="band")
        heart_sig = signal.filtfilt(b_c, a_c, g_signal)
        
        # Blinking: EAR signal (raw)
        ear_sig = np.array(ear_signals)
        
        # Respiration: nose vertical vertical displacements filtered between 0.1-1.0 Hz
        nose_arr = np.array(nose_pos)
        nose_y = nose_arr[:, 1]
        b_r, a_r = signal.butter(4, [0.1 / nyquist, 1.0 / nyquist], btype="band")
        resp_sig = signal.filtfilt(b_r, a_r, nose_y)
        
        # Head motion vertical velocity (vertical coordinate displacement)
        face_arr = np.array(face_pos)
        motion_sig = np.diff(face_arr[:, 1])
        # Pad motion signal to match the length of others
        motion_sig = np.append(motion_sig, motion_sig[-1]) 
        
        # 3. Compute 6 graph edge synchronization weights
        w_heart_ear = compute_max_cross_correlation(heart_sig, ear_sig)
        w_heart_resp = compute_max_cross_correlation(heart_sig, resp_sig)
        w_heart_motion = compute_max_cross_correlation(heart_sig, motion_sig)
        w_ear_resp = compute_max_cross_correlation(ear_sig, resp_sig)
        w_ear_motion = compute_max_cross_correlation(ear_sig, motion_sig)
        w_resp_motion = compute_max_cross_correlation(resp_sig, motion_sig)
        
        graph_edges = np.array([
            w_heart_ear,
            w_heart_resp,
            w_heart_motion,
            w_ear_resp,
            w_ear_motion,
            w_resp_motion
        ])
        
        return base_features, graph_edges

def main():
    print("=" * 70)
    print("PHYSIOLOGICAL CONSISTENCY GRAPH NETWORK (PCGN) PIPELINE")
    print("=" * 70)
    
    extractor = PCGNFeatureExtractor()
    
    X_base, X_graph, y_list, group_list = [], [], [], []
    
    for folder, label in [(REAL_FOLDER, 0), (FAKE_FOLDER, 1)]:
        videos = glob.glob(os.path.join(DATASET_PATH, folder, "*.mp4"))
        print(f"\n📂 Processing {len(videos)} {'real' if label == 0 else 'fake'} videos...")
        for video_path in tqdm(videos):
            res = extractor.extract_graph_features_from_video(video_path)
            if res is not None:
                base_f, graph_f = res
                X_base.append(base_f)
                X_graph.append(graph_f)
                y_list.append(label)
                group_list.append(extract_video_id(video_path))
                
    X_base = np.array(X_base)
    X_graph = np.array(X_graph)
    y = np.array(y_list)
    groups = np.array(group_list)
    
    print(f"\n✅ Extracted: {len(X_base)} videos")
    print(f"   Base features: {X_base.shape}")
    print(f"   Graph features: {X_graph.shape}")
    
    # Save extracted features to npz
    os.makedirs(os.path.join(PARENT_DIR, "results"), exist_ok=True)
    features_cache_path = os.path.join(PARENT_DIR, "results/graph_consistency_features.npz")
    np.savez(features_cache_path, 
             X_base=X_base, X_graph=X_graph, y=y, groups=groups)
    print(f"💾 Saved features to {features_cache_path}")
    
    # 10-fold cross validation comparison
    models_to_evaluate = {
        "Base Physiological (47 features)": X_base,
        "Graph Edge Synchronization (6 features)": X_graph,
        "Graph + Base Combined (53 features)": np.hstack([X_base, X_graph])
    }
    
    cv_results = {}
    gkf = GroupKFold(n_splits=10)
    
    for model_name, X_data in models_to_evaluate.items():
        print(f"\nEvaluating: {model_name}...")
        
        accs = []
        aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_data, y, groups), 1):
            X_train, X_val = X_data[train_idx], X_data[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=3,
                random_state=SEED,
                class_weight="balanced"
            )
            model.fit(X_train_scaled, y_train)
            
            prob = model.predict_proba(X_val_scaled)[:, 1]
            thresh, _ = find_optimal_threshold(y_val, prob)
            y_pred = (prob >= thresh).astype(int)
            
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else 0.5
            
            accs.append(acc)
            aucs.append(auc)
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_auc = np.mean(aucs)
        
        # Calculate t-distribution 95% CI
        sem = std_acc / np.sqrt(len(accs))
        ci = stats.t.interval(0.95, df=len(accs)-1, loc=mean_acc, scale=sem)
        
        # p-value against random chance
        t_stat, p_val = stats.ttest_1samp(accs, 0.50)
        
        print(f"   Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
        print(f"   95% CI:   [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
        print(f"   Mean AUC: {mean_auc:.4f}")
        print(f"   p-value:  {p_val:.4f}")
        
        cv_results[model_name] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "ci_accuracy": [ci[0], ci[1]],
            "mean_auc": mean_auc,
            "p_value": p_val,
            "accuracies": list(accs)
        }
        
    results_json_path = os.path.join(PARENT_DIR, "results/physiological_graph_results.json")
    with open(results_json_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"\n✅ PCGN Evaluation Complete. Saved to {results_json_path}")

if __name__ == "__main__":
    main()
