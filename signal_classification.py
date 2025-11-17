import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from feature_utils import extract_features_from_array


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


def plot_fft(signal, save_path="fft_spectrum.png"):
    n = len(signal)
    t = np.linspace(0, 1, n)
    xf = rfftfreq(n, 1/n)
    yf = np.abs(rfft(signal))

    plt.figure(figsize=(7, 4))
    plt.plot(xf, yf, color="orange")
    plt.title("Frequency Spectrum (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    print(f"Saved FFT spectrum plot as {save_path}")
    plt.show()


def main():
    X_raw, y = generate_signals(n_samples=1000, n_points=100)

    feature_list = [extract_features_from_array(sig) for sig in X_raw]
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

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix heatmap as confusion_matrix.png")
    plt.show()

    sample_indices = np.random.choice(len(X_test), size=6, replace=False)
    raw_indices = [list(X_test.index)[i] for i in sample_indices]

    # Multi-signal plot
    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    axs = axs.flatten()
    for ax, ridx, si in zip(axs, raw_indices, sample_indices):
        sig = X_raw[ridx]
        ax.plot(sig)
        ax.set_title(f"Actual: {y[si]} | Predicted: {y_pred[si]}")
        ax.set_xlabel("Sample #")
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    save_sample_csv(X_raw, y, "data/signals_sample.csv", n_save=50)

    # FFT on first plotted signal
    print("\nExtracted Features:")
    first_sig = X_raw[raw_indices[0]]
    feats = extract_features_from_array(first_sig)
    for k, v in feats.items():
        print(f"{k:25s}: {v}")

    # Plot FFT
    plot_fft(first_sig, "fft_spectrum.png")


if __name__ == "__main__":
    main()
