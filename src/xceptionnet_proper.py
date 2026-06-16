#!/usr/bin/env python3
"""
XceptionNet Baseline for Deepfake Detection (Frame-Averaging Classification)
"""

import os
import re
import cv2
import glob
import numpy as np
import tensorflow as tf
import random
import json
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("XCEPTIONNET FRAME-AVERAGING BASELINE")
print("=" * 70)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="./sdfvd_dataset/SDFVD")
args = parser.parse_args()

DATASET_PATH = args.dataset
IMG_SIZE = 128
SEQ_LENGTH = 20  # Number of frames to average per video
BATCH_SIZE = 4
EPOCHS = 20
INITIAL_LR = 0.0001
REAL_FOLDER = "videos_real"
FAKE_FOLDER = "videos_fake"
N_FOLDS = 10
MODEL_PATH = "face_landmarker.task"

class FaceProcessor:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_and_crop_face(self, frame, padding=0.2):
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)

        if results.face_landmarks:
            lms = results.face_landmarks[0]
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]
            
            x, y = min(xs) * w, min(ys) * h
            box_w, box_h = (max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h

            pad_x = int(box_w * padding)
            pad_y = int(box_h * padding)

            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(w, int(x + box_w + pad_x))
            y2 = min(h, int(y + box_h + pad_y))

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                return cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        return None

processor = FaceProcessor()

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQ_LENGTH:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames - 1, SEQ_LENGTH, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        face = processor.detect_and_crop_face(frame)
        if face is not None:
            frames.append(face)
            
    cap.release()
    if len(frames) == SEQ_LENGTH:
        return np.array(frames)
    return None

def extract_video_id(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r"(\d+)", basename)
    return match.group(1) if match else basename

def build_xceptionnet():
    # Load Xception base model pretrained on ImageNet
    base_model = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze convolutional layers
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)
    
    model = Model(base_model.input, output)
    model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Load Data
print("\n📂 Loading data...")
videos_list = []
y_list = []
groups_list = []

for folder, label in [(REAL_FOLDER, 0), (FAKE_FOLDER, 1)]:
    videos = glob.glob(os.path.join(DATASET_PATH, folder, "*.mp4"))
    print(f"   Processing {len(videos)} {'real' if label == 0 else 'fake'} videos...")
    for video_path in tqdm(videos):
        frames = extract_frames_from_video(video_path)
        if frames is not None:
            videos_list.append(frames)
            y_list.append(label)
            groups_list.append(extract_video_id(video_path))

X = np.array(videos_list)
y = np.array(y_list)
groups = np.array(groups_list)

print(f"\n✅ Data Loaded: {X.shape} samples (videos, frames, height, width, channels)")

# 10-Fold GroupKFold CV
gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
fold_accs = []
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    print(f"\n--- Fold {fold} ---")
    X_train_vids, X_val_vids = X[train_idx], X[val_idx]
    y_train_vids, y_val_vids = y[train_idx], y[val_idx]
    
    # Flatten video dimension to frames for 2D XceptionNet training
    # For train:
    num_train_vids, num_frames, h, w, c = X_train_vids.shape
    X_train_frames = X_train_vids.reshape(-1, h, w, c)
    y_train_frames = np.repeat(y_train_vids, num_frames)
    
    # For validation:
    num_val_vids = len(X_val_vids)
    X_val_frames = X_val_vids.reshape(-1, h, w, c)
    y_val_frames = np.repeat(y_val_vids, num_frames)
    
    # Data Augmentation (tf.data pipeline)
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.15)
        return image, label
        
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_frames, y_train_frames))
    train_ds = train_ds.shuffle(len(X_train_frames)).map(augment).batch(BATCH_SIZE * SEQ_LENGTH)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_frames, y_val_frames)).batch(BATCH_SIZE * SEQ_LENGTH)
    
    model = build_xceptionnet()
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, lr_reducer],
        verbose=0
    )
    
    # Predict frames and average to get video prediction
    video_probs = []
    for vid_frames in X_val_vids:
        # Predict probability for all 20 frames of the video
        frame_probs = model(vid_frames, training=False).numpy().flatten()
        video_probs.append(np.mean(frame_probs)) # Frame pooling
        
    video_probs = np.array(video_probs)
    video_preds = (video_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_val_vids, video_preds)
    auc = roc_auc_score(y_val_vids, video_probs) if len(np.unique(y_val_vids)) > 1 else 0.5
    
    fold_accs.append(acc)
    fold_aucs.append(auc)
    print(f"   Fold {fold}: Acc = {acc*100:.2f}%, AUC = {auc:.4f}")
    tf.keras.backend.clear_session()

mean_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)
mean_auc = np.mean(fold_aucs)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"📊 Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
print(f"📊 AUC:      {mean_auc:.4f}")

results = {
    "mean_accuracy": mean_acc,
    "std_accuracy": std_acc,
    "mean_auc": mean_auc,
    "accuracies": fold_accs
}

os.makedirs("results", exist_ok=True)
with open("results/xceptionnet_proper_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to results/xceptionnet_proper_results.json")
