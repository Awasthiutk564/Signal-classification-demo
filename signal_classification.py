import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from feature_utils import extract_features_from_array

def generate_signals(n_samples=1000, n_points=200):
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
            noise_level = np.random.uniform(0.05, 0.6)
            signal = base + noise_level * np.random.randn(len(t))
        X.append(signal)
        y.append(signal_type)
    return np.array(X), np.array(y)

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

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
    print(f"Saved sample CSV to {out_csv}")

def plot_and_save_time(signal, title, filename):
    plt.figure(figsize=(8,3))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_and_save_fft(signal, filename, fs=200):
    n = len(signal)
    xf = rfftfreq(n, 1/fs)
    yf = np.abs(rfft(signal))
    plt.figure(figsize=(8,3))
    plt.plot(xf, yf)
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def save_png_figure(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def main():
    X_raw, y = generate_signals(n_samples=1000, n_points=200)
    feature_list = [extract_features_from_array(sig) for sig in X_raw]
    feature_names = list(feature_list[0].keys())
    X = pd.DataFrame(feature_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print("\nConfusion Matrix:")
    print(cm)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Saved confusion_matrix.png")

    sample_indices = np.random.choice(len(X_test), size=6, replace=False)
    raw_indices = [list(X_test.index)[i] for i in sample_indices]

    fig, axs = plt.subplots(3,2, figsize=(10,6))
    axs = axs.flatten()
    for ax, ridx, si in zip(axs, raw_indices, sample_indices):
        sig = X_raw[ridx]
        ax.plot(sig)
        ax.set_title(f"Actual: {y[si]} | Pred: {y_pred[si]}")
        ax.set_xlabel("Sample #"); ax.set_ylabel("Amp")
    plt.tight_layout()
    plt.savefig("multi_signals.png")
    plt.close()
    print("Saved multi_signals.png")

    save_sample_csv(X_raw, y, "data/signals_sample.csv", n_save=50)

    first_sig = X_raw[raw_indices[0]]
    feats = extract_features_from_array(first_sig)
    print("\nExtracted features (example):")
    for k,v in feats.items():
        print(f"{k:25s}: {v}")

    # FFT before
    plot_and_save_fft(first_sig, "fft_before.png", fs=len(first_sig))

    # Low-pass filter (Butterworth)
    fs = len(first_sig)  # sampling frequency (since t in [0,1], fs = n_samples)
    cutoff = 12.0  # Hz - you can adjust
    filtered = butter_lowpass_filter(first_sig, cutoff=cutoff, fs=fs, order=4)

    # Save time-domain before/after and FFT after
    plot_and_save_time(first_sig, "Signal - Before Filtering", "time_before.png")
    plot_and_save_time(filtered, "Signal - After Low-pass Filtering", "time_after.png")
    plot_and_save_fft(filtered, "fft_after.png", fs=fs)

    # Save filter result CSV
    df_filtered = pd.DataFrame({"amplitude_filtered": filtered})
    df_filtered.to_csv("filtered_signal.csv", index=False)
    print("Saved filtered_signal.csv")

    # Save feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    fig_fi = plt.figure(figsize=(8,5))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color='skyblue')
    plt.xlabel("Importance"); plt.title("Feature Importance")
    plt.tight_layout()
    fig_fi.savefig("feature_importance.png")
    plt.close(fig_fi)
    print("Saved feature_importance.png")

if __name__ == "__main__":
    main()
