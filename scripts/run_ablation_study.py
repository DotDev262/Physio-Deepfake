#!/usr/bin/env python3
import os
import sys
import numpy as np
import json
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dynamic path resolution relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PARENT_DIR)

from src.physio_detection import find_optimal_threshold

SEED = 42

def main():
    print("🚀 Running Ablation Study on Physiological Features...")
    
    features_cache_path = os.path.join(PARENT_DIR, "results/learning_curve_features.npz")
    
    # Load pre-extracted features
    if not os.path.exists(features_cache_path):
        print(f"❌ Pre-extracted features file '{features_cache_path}' not found. Run run_learning_curves.py first!")
        sys.exit(1)
        
    data = np.load(features_cache_path)
    X = data["X_physio"]
    y = data["y_physio"]
    groups = data["groups_physio"]
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
    
    # Define indices of feature groups
    feature_groups = {
        "Full Model (All Features)": list(range(47)),
        "rPPG Only": list(range(0, 8)),
        "Blink Only": list(range(8, 19)),
        "Respiration Only": list(range(19, 26)),
        "Micro-movements Only": list(range(36, 47)),
        "Color Consistency Only": list(range(26, 36)),
        "Combined Biological (rPPG+Blink+Respiration+Micro)": list(range(0, 26)) + list(range(36, 47))
    }
    
    ablation_results = {}
    
    for name, indices in feature_groups.items():
        print(f"Evaluating: {name} ({len(indices)} features)...")
        X_sub = X[:, indices]
        
        accs = []
        gkf = GroupKFold(n_splits=10)
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_sub, y, groups), 1):
            X_train, X_val = X_sub[train_idx], X_sub[val_idx]
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
            
            accs.append(accuracy_score(y_val, y_pred))
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"   Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
        ablation_results[name] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "num_features": len(indices)
        }
        
    ablation_json_path = os.path.join(PARENT_DIR, "results/ablation_study_results.json")
    with open(ablation_json_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
        
    print(f"\n✅ Ablation Study Complete. Saved results to {ablation_json_path}")

if __name__ == "__main__":
    main()
