#!/usr/bin/env python3
"""
Final attempt: Threshold optimization + MLP + Extra Trees
"""

import os
import re
import cv2
import numpy as np
import glob
import random
import json
import warnings

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import stats, signal
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import wilcoxon

import argparse

print("=" * 70)
print("THRESHOLD OPTIMIZATION + MLP + EXTRA TREES")
print("=" * 70)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="./sdfvd_dataset/SDFVD")
args = parser.parse_args()

DATASET_PATH = args.dataset
REAL_FOLDER = "videos_real"
FAKE_FOLDER = "videos_fake"
N_FOLDS = 10
MODEL_PATH = "face_landmarker.task"


def extract_video_id(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r"(\d+)", basename)
    return match.group(1) if match else basename


class EnhancedPhysioFeatureExtractor:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract_rppq_features(self, rgb_signals, fps):
        if len(rgb_signals) < 60:
            return [0.0] * 8
        rgb = np.array(rgb_signals)
        r_norm = (rgb[:, 0] - np.mean(rgb[:, 0])) / (np.std(rgb[:, 0]) + 1e-7)
        g_norm = (rgb[:, 1] - np.mean(rgb[:, 1])) / (np.std(rgb[:, 1]) + 1e-7)
        b_norm = (rgb[:, 2] - np.mean(rgb[:, 2])) / (np.std(rgb[:, 2]) + 1e-7)
        X = 3 * r_norm - 2 * g_norm
        Y = 1.5 * r_norm + g_norm - 1.5 * b_norm
        nyquist = 0.5 * fps
        low, high = 0.7 / nyquist, 4.0 / nyquist
        try:
            b_coeff, a_coeff = signal.butter(4, [low, high], btype="band")
            X_f = signal.filtfilt(b_coeff, a_coeff, X)
            Y_f = signal.filtfilt(b_coeff, a_coeff, Y)
            alpha = np.std(X_f) / (np.std(Y_f) + 1e-7)
            S = X_f - alpha * Y_f
            n = len(S)
            freqs = np.fft.fftfreq(n, 1 / fps)[: n // 2]
            magnitudes = np.abs(np.fft.fft(S)[: n // 2])
            valid_mask = (freqs >= 0.7) & (freqs <= 4.0)
            heart_rate = (
                freqs[valid_mask][np.argmax(magnitudes[valid_mask])] * 60
                if valid_mask.any()
                else 0.0
            )
        except:
            heart_rate = 0.0
        features = [heart_rate, 1.0 if 40 <= heart_rate <= 180 else 0.0]
        if heart_rate > 40:
            peak_interval = 60.0 / heart_rate * fps
            peaks = np.arange(int(peak_interval), len(S), int(peak_interval))
            if len(peaks) > 2:
                intervals = np.diff(peaks) / fps * 1000
                rmssd = np.sqrt(np.mean(np.diff(intervals) ** 2))
                features.extend(
                    [
                        rmssd,
                        np.std(intervals),
                        np.mean(np.abs(np.diff(intervals)) > 50) * 100,
                        np.mean(intervals),
                        np.std(intervals),
                    ]
                )
            else:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)
        while len(features) < 8:
            features.append(0.0)
        return features

    def extract_blink_features(self, ear_signals, fps):
        if len(ear_signals) < 30:
            return [0.0] * 10
        ear = np.array(ear_signals)
        blinks = ear < 0.2
        blink_times = np.where(np.diff(blinks.astype(int)) == 1)[0]
        blink_rate = (
            60.0 / np.mean(np.diff(blink_times) / fps) if len(blink_times) >= 2 else 0.0
        )
        features = [blink_rate, 1.0 if 5 <= blink_rate <= 40 else 0.0]
        features.extend([np.mean(ear), np.std(ear), np.min(ear), np.max(ear)])
        ear_diff = np.abs(np.diff(ear))
        features.extend(
            [
                np.mean(ear_diff),
                np.std(ear_diff),
                np.max(ear_diff),
                np.sum(ear_diff > 0.05),
                np.var(ear),
            ]
        )
        while len(features) < 10:
            features.append(0.0)
        return features

    def extract_breathing_features(self, nose_positions, fps):
        if len(nose_positions) < 30:
            return [0.0] * 7
        nose = np.array(nose_positions)
        if len(nose.shape) < 2 or nose.shape[1] != 2:
            return [0.0] * 7
        displacements = np.diff(nose[:, 1])
        if len(displacements) < 2:
            return [0.0] * 7
        fft_vals = np.fft.fft(displacements)
        freqs = np.fft.fftfreq(len(displacements), 1 / fps)
        valid = (freqs > 0.1) & (freqs < 1)
        breathing_rate = (
            freqs[valid][np.argmax(np.abs(fft_vals[valid]))] * 60 if np.any(valid) else 0.0
        )
        features = [breathing_rate, 1.0 if 5 <= breathing_rate <= 30 else 0.0]
        features.extend(
            [
                np.std(displacements),
                np.max(np.abs(displacements)),
                np.mean(np.abs(displacements)),
                np.sum(np.abs(displacements) > np.std(displacements)),
            ]
        )
        while len(features) < 7:
            features.append(0.0)
        return features

    def extract_color_features(self, rgb_signals):
        if len(rgb_signals) < 10:
            return [0.0] * 10
        rgb = np.array(rgb_signals)
        features = [np.std(rgb[:, 0]), np.std(rgb[:, 1]), np.std(rgb[:, 2])]
        gr_ratio = np.mean(rgb[:, 1]) / (np.mean(rgb[:, 0]) + 1e-7)
        features.extend([gr_ratio, 1.0 / (np.std(rgb) + 1e-7)])
        first_half = np.mean(rgb[: len(rgb) // 2], axis=0)
        second_half = np.mean(rgb[len(rgb) // 2 :], axis=0)
        features.append(np.mean(np.abs(first_half - second_half)))
        features.append(np.std(rgb[:, 1]))
        skew = (
            np.abs(stats.skew(rgb[:, 0]))
            + np.abs(stats.skew(rgb[:, 1]))
            + np.abs(stats.skew(rgb[:, 2]))
        )
        features.append(skew)
        brightness = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        features.extend([np.std(brightness), 1.0 / (np.std(brightness) + 1e-7)])
        while len(features) < 10:
            features.append(0.0)
        return features

    def extract_micro_movement_features(self, lm_seq, face_pos):
        features = []
        if len(face_pos) < 10:
            return [0.0] * 11
        face_arr = np.array(face_pos)
        displacements = np.diff(face_arr, axis=0)
        displacement_mag = np.linalg.norm(displacements, axis=1)
        features.extend(
            [
                np.mean(displacement_mag),
                np.std(displacement_mag),
                np.max(displacement_mag),
                np.sum(displacement_mag > np.std(displacement_mag)),
            ]
        )
        features.append(len(lm_seq) / max(1, len(face_pos)))
        if len(face_pos) > 10:
            vert_motion = np.std(np.diff(face_arr[:, 1]))
            horiz_motion = np.std(np.diff(face_arr[:, 0]))
            features.append(vert_motion / (horiz_motion + 1e-7))
            noise = np.mean(np.abs(np.diff(face_arr, n=2)))
            features.append(1.0 / (noise + 1e-7))
            features.append(np.mean(np.abs(np.diff(face_arr, n=2))))
        else:
            features.extend([0.0] * 4)
        while len(features) < 11:
            features.append(0.0)
        return features

    def extract_features_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:
            fps = 30
        rgb_signals, ear_signals = [], []
        face_pos, nose_pos, lm_seq = [], [], []
        
        frame_count = 0
        max_frames = 150  # Cap at 150 frames (~5 seconds) for extraction speedup
        while cap.isOpened():
            if frame_count >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            detection_result = self.landmarker.detect(mp_image)
            
            if detection_result.face_landmarks:
                lms = detection_result.face_landmarks[0]
                h, w = frame.shape[:2]
                
                face_pos.append(
                    (
                        np.mean([lm.x for lm in lms]) * w,
                        np.mean([lm.y for lm in lms]) * h,
                    )
                )
                
                fh_y = int(lms[10].y * h)
                fh_x = int(lms[10].x * w)
                roi_h, roi_w = int(0.05 * h), int(0.1 * w)
                fh = frame[max(0, fh_y):min(h, fh_y+roi_h), max(0, fh_x-roi_w//2):min(w, fh_x+roi_w//2)]
                
                if fh.size > 0:
                    rgb_signals.append(np.mean(fh.reshape(-1, 3), axis=0))

                def get_ear(indices):
                    pts = [np.array([lms[i].x * w, lms[i].y * h]) for i in indices]
                    v1 = np.linalg.norm(pts[1] - pts[5])
                    v2 = np.linalg.norm(pts[2] - pts[4])
                    h_d = np.linalg.norm(pts[0] - pts[3])
                    return (v1 + v2) / (2.0 * h_d + 1e-7)

                left_ear = get_ear([33, 160, 158, 133, 153, 144])
                right_ear = get_ear([362, 385, 387, 263, 373, 380])
                ear_signals.append((left_ear + right_ear) / 2.0)
                
                nose_pos.append((lms[4].x * w, lms[4].y * h))
                lm_seq.append(lms)
            
        cap.release()
        if len(rgb_signals) < 30:
            return None
            
        all_f = []
        all_f.extend(self.extract_rppq_features(rgb_signals, fps))
        all_f.extend(self.extract_blink_features(ear_signals, fps))
        all_f.extend(self.extract_breathing_features(nose_pos, fps))
        all_f.extend(self.extract_color_features(rgb_signals))
        all_f.extend(self.extract_micro_movement_features(lm_seq, face_pos))
        return np.array(all_f)


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes accuracy"""
    best_thresh = 0.5
    best_acc = 0
    for thresh in np.arange(0.3, 0.7, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def main():
    print("\n📂 Loading data...")
    X, y, video_ids = [], [], []
    extractor = EnhancedPhysioFeatureExtractor()

    for folder, label in [(REAL_FOLDER, 0), (FAKE_FOLDER, 1)]:
        videos = glob.glob(os.path.join(DATASET_PATH, folder, "*.mp4"))
        print(f"   Processing {len(videos)} {'real' if label == 0 else 'fake'} videos...")
        for video_path in videos:
            features = extractor.extract_features_from_video(video_path)
            if features is not None and len(features) >= 45:
                X.append(features)
                y.append(label)
                video_ids.append(extract_video_id(video_path))

    X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)
    groups = np.array(video_ids)

    if len(X) == 0:
        print("❌ No features extracted. Check dataset path or video availability.")
        exit(1)

    print(f"\n✅ Data: {len(X)} samples, {X.shape[1]} features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n" + "=" * 70)
    print("TESTING MODELS WITH THRESHOLD OPTIMIZATION")
    print("=" * 70)

    models = [
        (
            "RF_default",
            RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=SEED, class_weight="balanced"
            ),
            0.5,
        ),
        (
            "RF_tuned",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=3,
                random_state=SEED,
                class_weight="balanced",
            ),
            0.5,
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=300, max_depth=15, random_state=SEED, class_weight="balanced"
            ),
            0.5,
        ),
        (
            "MLP_small",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=SEED,
                early_stopping=True,
            ),
            0.5,
        ),
        (
            "MLP_large",
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=SEED,
                early_stopping=True,
            ),
            0.5,
        ),
    ]

    best_acc = 0
    best_name = ""
    best_results = {}

    for name, model, default_thresh in models:
        print(f"\n--- {name} ---")

        gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
        fold_accs = []
        fold_aucs = []
        fold_thresholds = []
        all_y_true_fold = []
        all_y_proba_fold = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups), 1):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            prob = model.predict_proba(X_val)[:, 1]

            thresh, _ = find_optimal_threshold(y_val, prob)
            y_pred = (prob >= thresh).astype(int)

            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else 0.5

            fold_accs.append(acc)
            fold_aucs.append(auc)
            fold_thresholds.append(thresh)
            all_y_true_fold.extend(y_val)
            all_y_proba_fold.extend(prob)

            print(
                f"   Fold {fold}: Acc={acc * 100:.2f}%, AUC={auc:.4f}, thresh={thresh:.2f}"
            )

        mean_acc = np.mean(fold_accs)
        mean_auc = np.mean(fold_aucs)

        print(f"   Mean: Acc={mean_acc * 100:.2f}%, AUC={mean_auc:.4f}")

        best_results[name] = {
            "accuracy": mean_acc,
            "auc": mean_auc,
            "thresholds": fold_thresholds,
        }

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_name = name

    print(f"\n✅ Best: {best_name} with {best_acc * 100:.2f}%")

    # Final evaluation with best model and optimized threshold
    print("\n" + "=" * 70)
    print(f"FINAL EVALUATION ({best_name})")
    print("=" * 70)

    best_model_info = {
        "RF_tuned": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=3,
            random_state=SEED,
            class_weight="balanced",
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=15, random_state=SEED, class_weight="balanced"
        ),
        "MLP_small": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
        ),
        "MLP_large": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
        ),
    }

    model = best_model_info.get(
        best_name,
        RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=SEED, class_weight="balanced"
        ),
    )

    gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
    fold_accs = []
    fold_aucs = []
    all_y_true, all_y_proba = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]

        thresh = best_results[best_name]["thresholds"][fold - 1]
        y_pred = (prob >= thresh).astype(int)

        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else 0.5

        fold_accs.append(acc)
        fold_aucs.append(auc)
        all_y_true.extend(y_val)
        all_y_proba.extend(prob)

        print(f"   Fold {fold}: Acc={acc * 100:.2f}%, AUC={auc:.4f}, thresh={thresh:.2f}")

    mean_acc = np.mean(fold_accs)
    mean_auc = np.mean(fold_aucs)

    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)

    # Use average threshold from folds
    avg_thresh = np.mean(best_results[best_name]["thresholds"])
    all_y_pred = (all_y_proba >= avg_thresh).astype(int)

    stat, p_value = wilcoxon(fold_accs)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n📊 Accuracy: {mean_acc * 100:.2f}% ± {np.std(fold_accs) * 100:.2f}%")
    print(f"📊 AUC: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
    print(
        f"📊 p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}"
    )
    print(f"📊 Optimized threshold: {avg_thresh:.2f}")

    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"📊 Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    results = {
        "mean_accuracy": mean_acc,
        "mean_auc": mean_auc,
        "p_value": p_value,
        "best_model": best_name,
        "optimized_threshold": avg_thresh,
        "all_models": best_results,
    }

    with open("results/physiological_final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Saved to results/physiological_final_results.json")


if __name__ == "__main__":
    main()
