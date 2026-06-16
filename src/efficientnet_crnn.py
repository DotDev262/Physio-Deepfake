#!/usr/bin/env python3
"""
EfficientNet-B0 + Stacked BiLSTM + Attention for Deepfake Detection
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
from tensorflow.keras.layers import (
    TimeDistributed,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.applications import EfficientNetB0
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
print("EFFICIENTNET-B0 + BILSTM + ATTENTION BASELINE")
print("=" * 70)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="./sdfvd_dataset/SDFVD")
args = parser.parse_args()

DATASET_PATH = args.dataset
IMG_SIZE = 128
SEQ_LENGTH = 20
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

def build_efficientnet_crnn():
    # Spatial Backbone
    backbone = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False  # Freeze backbone features
    
    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(512, activation="relu")(x)
    backbone_model = Model(backbone.input, x)
    
    # Recurrent Model
    video_input = Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    encoded_frames = TimeDistributed(backbone_model)(video_input)
    
    # Stacked BiLSTM
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(encoded_frames)
    lstm_2 = Bidirectional(LSTM(64, return_sequences=True))(lstm_1)
    
    # Attention Layer
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_2, lstm_2)
    attn = LayerNormalization()(attn)
    
    # Temporal pooling / final classification
    flat = GlobalAveragePooling1D()(attn) # Mean pool over temporal axis
    out = Dense(64, activation="relu")(flat)
    out = Dropout(0.4)(out)
    output = Dense(1, activation="sigmoid")(out)
    
    model = Model(video_input, output)
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

print(f"\n✅ Data Loaded: {X.shape} samples")

# 10-Fold GroupKFold CV
gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
fold_accs = []
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    print(f"\n--- Fold {fold} ---")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Data Augmentation (tf.data pipeline)
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label
        
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train)).map(augment).batch(BATCH_SIZE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    
    model = build_efficientnet_crnn()
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, lr_reducer],
        verbose=0
    )
    
    probs = model(X_val, training=False).numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
    
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
with open("results/efficientnet_crnn_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to results/efficientnet_crnn_results.json")
