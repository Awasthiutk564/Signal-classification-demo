# -----------------------------------------------------------
# DAY 6 — Signal Classification Project
# Features:
# - Sine / Square / Sawtooth / Noisy signals
# - Time + Frequency domain features
# - RandomForest model
# - Accuracy + Classification Report + Confusion Matrix (Heatmap)
# - Multi-signal plotting
# - CSV Export
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from feature_utils import extract_features_from_array

# -----------------------------------------------------------
# Signal Generator
# -----------------------------------------------------------
def generate_signals(n_samples=1000, n_points=100):
    X = []
    y = []

    for _ in range(n_samples):
        signal_type = np.random.choice(['sine', 'square', 'sawtooth', 'noisy'])
        t = np.linspace(0, 1, n_points)

        if signal_type == 'sine':
            signal = np.sin(2 * np.pi * 5 * t)

        elif signal_type == 'square':
            signal = np.sign(np.sin(2 * np.pi * 5 * t))

        elif signal_type == 'sawtooth':
            signal = 2 * (t - np.floor(t + 0.5))

        elif signal_type == 'noisy':
            base = np.sin(2 * np.pi * 5 * t)
            noise_level = np.random.uniform(0.05, 0.5)
            signal = base + noise_level * np.random.randn(len(t))

        X.append(signal)
        y.append(signal_type)

    return np.array(X), np.array(y)

# -----------------------------------------------------------
# Save RAW Signals to CSV
# -----------------------------------------------------------
def save_sample_csv(X_raw, y, out_csv="data/signals_sample.csv", n_save=50):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []
    n_save = min(n_save, len(X_raw))

    for i in range(n_save):
        row = {"label": y[i]}
        for j, val in enumerate(X_raw[i]):
            row[f"s{j}"] = float(val)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"📁 Saved sample CSV to: {out_csv}")

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():

    # 1) Generate Dataset
    X_raw, y = generate_signals(n_samples=1000, n_points=100)

    # 2) Feature Extraction
    feature_list = [extract_features_from_array(sig) for sig in X_raw]
    X = pd.DataFrame(feature_list)

    # 3) Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Train Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 5) Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy (Day 6): {acc*100:.2f}%\n")

    # 6) Classification Report
    print("📄 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7) Confusion Matrix + Heatmap
    print("🧾 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(cm)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap (Day 6)")
    plt.show()

    # 8) Multi-signal plotting (6 random signals)
    sample_indices = np.random.choice(len(X_test), size=6, replace=False)
    raw_indices = [list(X_test.index)[i] for i in sample_indices]

    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    axs = axs.flatten()

    for ax, ridx, si in zip(axs, raw_indices, sample_indices):
        sig = X_raw[ridx]
        ax.plot(sig)
        ax.set_title(f"Actual: {y[si]} | Predicted: {y_pred[si]}")
        ax.set_xlabel("Sample #")
        ax.set_ylabel("Amp")

    plt.tight_layout()
    plt.show()

    # 9) Save first 50 samples as CSV
    save_sample_csv(X_raw, y, "data/signals_sample.csv", n_save=50)

    # 10) Print features of first plotted signal
    print("\n📌 Extracted Features for 1 sample:")
    feats = extract_features_from_array(X_raw[raw_indices[0]])
    for k, v in feats.items():
        print(f"{k:25s}: {v}")

# -----------------------------------------------------------
if __name__ == "__main__":
    main()
