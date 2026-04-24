#!/usr/bin/env python3
"""
Traditional ML on Visual Features with Proper GroupKFold + Statistical Rigor
Features: Facial landmarks + color histograms
Classifiers: Random Forest, SVM, Ensemble
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
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from collections import Counter

import argparse

print("=" * 70)
print("TRADITIONAL ML (VISUAL FEATURES) - PROPER GROUPKFOLD")
print("=" * 70)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="./sdfvd_dataset/SDFVD")
args = parser.parse_args()

DATASET_PATH = args.dataset
IMG_SIZE = 128
REAL_FOLDER = "videos_real"
FAKE_FOLDER = "videos_fake"
N_FOLDS = 10
MODEL_PATH = "face_landmarker.task"

print(f"\n📊 Configuration:")
print(f"   Dataset: {DATASET_PATH}")
print(f"   Seed: {SEED}")


def extract_video_id(filepath):
    """Extract numeric ID from filename (e.g., 'v10.mp4' -> '10', 'vs10.mp4' -> '10')
    This ensures real and fake pairs of the same video are in the SAME fold."""
    basename = os.path.basename(filepath)
    match = re.search(r"(\d+)", basename)
    return match.group(1) if match else basename


class LandmarkExtractor:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract_landmark_features(self, frame):
        """Extract facial landmark features using Tasks API"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks:
            return np.zeros(11)

        landmarks = results.face_landmarks[0]
        h, w = frame.shape[:2]
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        left_eye = points[33]
        right_eye = points[263]
        nose = points[4]
        mouth_left = points[61]
        mouth_right = points[291]
        chin = points[152]

        eye_distance = np.linalg.norm(left_eye - right_eye)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        nose_to_chin = np.linalg.norm(nose - chin)

        left_eye_top = points[159]
        left_eye_bottom = points[145]
        left_eye_left = points[33]
        left_eye_right = points[133]

        right_eye_top = points[386]
        right_eye_bottom = points[374]
        right_eye_right = points[263]
        right_eye_left = points[362]

        left_ear = np.linalg.norm(left_eye_top - left_eye_bottom) / (
            np.linalg.norm(left_eye_left - left_eye_right) + 1e-7
        )
        right_ear = np.linalg.norm(right_eye_top - right_eye_bottom) / (
            np.linalg.norm(right_eye_right - right_eye_left) + 1e-7
        )

        face_width = np.linalg.norm(points[234] - points[454])
        face_height = np.linalg.norm(points[10] - points[152])

        return np.array(
            [
                eye_distance / face_width,
                mouth_width / face_width,
                nose_to_chin / face_height,
                left_ear,
                right_ear,
                face_width / face_height,
                np.std([lm.x for lm in landmarks]),
                np.std([lm.y for lm in landmarks]),
                mouth_width / nose_to_chin,
                eye_distance / nose_to_chin,
                (left_ear + right_ear) / 2,
            ]
        )


landmark_extractor = LandmarkExtractor()


def extract_color_histogram(frame):
    """Extract color histogram features from frame"""
    features = []
    for i in range(3):
        hist = cv2.calcHist([frame], [i], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
    return np.array(features)


def process_video(video_path):
    """Extract visual features from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 10:
        cap.release()
        return None

    sample_indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)

    all_landmarks = []
    all_colors = []

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        landmarks = landmark_extractor.extract_landmark_features(frame)
        if landmarks.sum() > 0:
            all_landmarks.append(landmarks)

        color_hist = extract_color_histogram(frame)
        all_colors.append(color_hist)

    cap.release()

    if len(all_landmarks) < 3:
        return None

    # Enhanced temporal aggregation
    landmark_features = np.mean(all_landmarks, axis=0)
    landmark_std = np.std(all_landmarks, axis=0)
    landmark_min = np.min(all_landmarks, axis=0)
    landmark_max = np.max(all_landmarks, axis=0)
    landmark_range = landmark_max - landmark_min

    landmark_diffs = np.diff(all_landmarks, axis=0)
    landmark_diff_mean = (
        np.mean(np.abs(landmark_diffs), axis=0)
        if len(landmark_diffs) > 0
        else np.zeros_like(landmark_features)
    )

    color_features = np.mean(all_colors, axis=0)
    color_std = np.std(all_colors, axis=0)
    color_min = np.min(all_colors, axis=0)
    color_max = np.max(all_colors, axis=0)
    color_range = color_max - color_min

    features = np.concatenate(
        [
            landmark_features,
            landmark_std,
            landmark_min,
            landmark_max,
            landmark_range,
            landmark_diff_mean,
            color_features,
            color_std,
            color_min,
            color_max,
            color_range,
        ]
    )
    return features


def load_data():
    """Load data without augmentation"""
    X = []
    y = []
    video_ids = []

    print("\n📂 Loading videos...")

    real_videos = glob.glob(os.path.join(DATASET_PATH, REAL_FOLDER, "*.mp4"))
    print(f"   Loading {len(real_videos)} Real Videos...")
    for video_path in real_videos:
        features = process_video(video_path)
        if features is not None:
            X.append(features)
            y.append(0)
            video_ids.append(extract_video_id(video_path))

    fake_videos = glob.glob(os.path.join(DATASET_PATH, FAKE_FOLDER, "*.mp4"))
    print(f"   Loading {len(fake_videos)} Fake Videos...")
    for video_path in fake_videos:
        features = process_video(video_path)
        if features is not None:
            X.append(features)
            y.append(1)
            video_ids.append(extract_video_id(video_path))

    X = np.array(X)
    y = np.array(y)
    video_ids = np.array(video_ids)

    class_counts = Counter(y)
    print(f"\n✅ Dataset loaded:")
    print(f"   Total: {len(y)} | Real: {class_counts[0]} | Fake: {class_counts[1]}")
    print(f"   Feature dim: {X.shape[1]}")

    return X, y, video_ids


def run_cv_for_classifier(clf_name, clf, X, y, groups):
    """Run GroupKFold for a classifier"""
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_aucs = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_prob = []

    print(f"\n   Training {clf_name}...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf_copy = (
            type(clf)(**clf.get_params()) if hasattr(clf, "get_params") else type(clf)()
        )

        if hasattr(clf, "random_state"):
            clf_copy.set_params(random_state=SEED + fold)

        clf_copy.fit(X_train_scaled, y_train)

        y_pred = clf_copy.predict(X_val_scaled)
        y_pred_prob = (
            clf_copy.predict_proba(X_val_scaled)[:, 1]
            if hasattr(clf_copy, "predict_proba")
            else y_pred
        )

        fold_accuracies.append(accuracy_score(y_val, y_pred))
        fold_precisions.append(precision_score(y_val, y_pred, zero_division=0))
        fold_recalls.append(recall_score(y_val, y_pred, zero_division=0))
        try:
            fold_aucs.append(roc_auc_score(y_val, y_pred_prob))
        except:
            fold_aucs.append(0.5)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_pred_prob.extend(
            y_pred_prob if isinstance(y_pred_prob, list) else y_pred_prob.tolist()
        )

        print(
            f"      Fold {fold + 1}: Acc={fold_accuracies[-1] * 100:.2f}% | AUC={fold_aucs[-1]:.4f}"
        )

    return {
        "accuracies": fold_accuracies,
        "precisions": fold_precisions,
        "recalls": fold_recalls,
        "aucs": fold_aucs,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
        "all_y_pred_prob": all_y_pred_prob,
    }


def compute_stats(results):
    """Compute statistical rigor metrics"""
    accuracies = results["accuracies"]
    precisions = results["precisions"]
    recalls = results["recalls"]
    aucs = results["aucs"]

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    t_acc, p_acc = stats.ttest_1samp(accuracies, 0.5)
    t_auc, p_auc = stats.ttest_1samp(aucs, 0.5)

    ci_acc = stats.t.interval(
        0.95, len(accuracies) - 1, loc=mean_acc, scale=stats.sem(accuracies)
    )
    ci_auc = stats.t.interval(0.95, len(aucs) - 1, loc=mean_auc, scale=stats.sem(aucs))

    return {
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "mean_precision": float(np.mean(precisions)),
        "std_precision": float(np.std(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "mean_auc": float(mean_auc),
        "std_auc": float(std_auc),
        "ci_accuracy": [float(ci_acc[0]), float(ci_acc[1])],
        "ci_auc": [float(ci_auc[0]), float(ci_auc[1])],
        "p_value_accuracy": float(p_acc),
        "p_value_auc": float(p_auc),
        "significant": bool(p_acc < 0.05),
    }


# Main execution
print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

X, y, video_ids = load_data()

if len(X) == 0:
    print("❌ No data loaded!")
    exit(1)

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, random_state=SEED
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED
    ),
    "SVM (Linear)": SVC(kernel="linear", C=1.0, probability=True, random_state=SEED),
}

print("\n" + "=" * 70)
print(f"{N_FOLDS}-FOLD GROUPKFOLD CV WITH STATISTICAL RIGOR")
print("=" * 70)

all_results = {}

for clf_name, clf in classifiers.items():
    cv_results = run_cv_for_classifier(clf_name, clf, X, y, video_ids)
    stats_results = compute_stats(cv_results)
    all_results[clf_name] = stats_results

    print(f"\n📊 {clf_name}:")
    print(
        f"   Accuracy: {stats_results['mean_accuracy'] * 100:.2f}% ± {stats_results['std_accuracy'] * 100:.2f}%"
    )
    print(
        f"   AUC:      {stats_results['mean_auc']:.4f} ± {stats_results['std_auc']:.4f}"
    )
    print(
        f"   p-value:  {stats_results['p_value_accuracy']:.6f} {'**' if stats_results['significant'] else ''}"
    )
    print(
        f"   95% CI:   [{stats_results['ci_accuracy'][0] * 100:.2f}%, {stats_results['ci_accuracy'][1] * 100:.2f}%]"
    )

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\n{'Model':<20} {'Accuracy':<20} {'AUC':<15} {'p-value':<12} {'Sig?'}")
print("-" * 75)
for name, stats in all_results.items():
    print(
        f"{name:<20} {stats['mean_accuracy'] * 100:.2f}% ± {stats['std_accuracy'] * 100:.2f}%   {stats['mean_auc']:.4f} ± {stats['std_auc']:.4f}   {stats['p_value_accuracy']:.6f}   {'Yes' if stats['significant'] else 'No'}"
    )

# Save results
final_results = {
    "model_type": "Traditional ML (Visual Features)",
    "features": "Facial landmarks + color histograms",
    "data_split": f"GroupKFold ({N_FOLDS}-fold)",
    "seed": SEED,
    "results": all_results,
}

with open("results/traditional_ml_proper_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\n✅ Results saved to results/traditional_ml_proper_results.json")
print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
