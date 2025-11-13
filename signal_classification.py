# -----------------------------
# Signal Classification (Day 5)
# - Feature extraction (from feature_utils)
# - Sine / Square / Sawtooth / Noisy
# - Plot multiple random test signals in one figure
# - Save a sample CSV of raw signals to data/signals_sample.csv
# -----------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from feature_utils import extract_features_from_array

# ---------- Signal generator ----------
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

# ---------- Save raw sample CSV ----------
def save_sample_csv(X_raw, y, out_csv="data/signals_sample.csv", n_save=50):
    """
    Save n_save rows (raw samples) to CSV with columns: label, s0, s1, ..., sN
    """
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
    print(f"Saved sample CSV to {out_csv} (rows: {len(df)})")

# ---------- Main flow ----------
def main():
    # 1) Generate
    X_raw, y = generate_signals(n_samples=1000, n_points=100)

    # 2) Extract features and train on features (as before)
    feature_list = [extract_features_from_array(sig) for sig in X_raw]
    X = pd.DataFrame(feature_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy (Day 5): {acc*100:.2f}%\n")

    # 3) Save a sample CSV of raw signals (first 50)
    save_sample_csv(X_raw, y, out_csv="data/signals_sample.csv", n_save=50)

    # 4) Plot multiple random test signals in one figure
    #    We'll pick 6 random test indices and show them tiled
    sample_indices = np.random.choice(len(X_test), size=6, replace=False)
    # map X_test indices back to raw array indices
    raw_indices = [list(X_test.index)[i] for i in sample_indices]
    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    axs = axs.flatten()
    for ax_i, ridx, si in zip(axs, raw_indices, sample_indices):
        sig = X_raw[ridx]
        ax_i.plot(sig)
        ax_i.set_title(f"Actual: {y[si]} | Predicted: {y_pred[si]}")
        ax_i.set_xlabel("Sample #")
        ax_i.set_ylabel("Amp")
    plt.tight_layout()
    plt.show()

    # 5) Print features for one of the plotted signals
    chosen = raw_indices[0]
    feats = extract_features_from_array(X_raw[chosen])
    print("\n📌 Extracted features for one sample plotted:")
    for k, v in feats.items():
        print(f"{k:25s}: {v}")

if __name__ == "__main__":
    main()
