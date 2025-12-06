# ⚡ Signal Classification System

A complete end-to-end project that classifies synthetic signals (sine, square, sawtooth, noisy) using Machine Learning and Digital Signal Processing, and exposes an interactive Streamlit web app for demo and testing.

---

## Features
- Generate signals: sine, square, sawtooth, noisy (adjust frequency & noise)
- Time-domain visualization and FFT (frequency spectrum)
- Butterworth low-pass filtering (before / after)
- Feature extraction (time + frequency features)
- Random Forest classifier + feature importance & confusion matrix
- Upload CSV → backend prediction → download filtered CSV
- Download raw/filtered CSV and waveform PNG
- Oscilloscope-style animation in web app
- Deployed Streamlit app (link in repo)

---

## Files
- `app.py` — Streamlit web dashboard (interactive demo)
- `signal_classification.py` — ML training + DSP script; generates PNG/CSV artifacts
- `feature_utils.py` — feature extraction helper
- `requirements.txt` — required Python packages
- `README.md` — this file
- `data/` — generated CSVs (auto)
- `*.png`, `*.gif`, `*.pdf` — artifacts (auto)

---

## Quick setup (local)
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
Run ML script (generates artifacts):

python signal_classification.py

Run the web app:

python -m streamlit run app.py
How to use the app (short)

Use sidebar to select signal type, frequency, noise and cutoff.

Preview time-domain and FFT (before/after filtering).

Upload your own CSV (time,amplitude) to predict and download filtered result.

Download waveform PNG / CSV from buttons.

Deployment

Frontend is Streamlit; deploy on Streamlit Cloud (share.streamlit.io).

Push repo to GitHub, then create new app on Streamlit Cloud selecting app.py.

Notes

This project is demo-ready. For very large real-world signals, adjust sampling/settings.

All generated files (PNG/CSV) are reproducible by running signal_classification.py.

---

## 2) **feature_utils.py** — copy entire file
```python
# feature_utils.py
import numpy as np
from scipy import stats
from scipy.fft import rfft

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def spectral_energy(x):
    # total energy in positive FFT bins
    yf = np.abs(rfft(x))
    return np.sum(yf**2) / len(yf)

def band_energy_ratios(x, bands=4):
    # split FFT magnitude into equal-width bands and return normalized energies
    yf = np.abs(rfft(x))
    N = len(yf)
    band_size = N // bands
    ratios = []
    total = np.sum(yf) + 1e-12
    for i in range(bands):
        start = i*band_size
        end = (i+1)*band_size if i < bands-1 else N
        ratios.append(np.sum(yf[start:end]) / total)
    return ratios

def extract_features_from_array(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {
            "mean": 0.0, "std": 0.0, "min":0.0, "max":0.0,
            "rms":0.0, "skew":0.0, "kurtosis":0.0,
            "spectral_energy":0.0, "band0":0.0, "band1":0.0, "band2":0.0, "band3":0.0
        }
    feat = {}
    feat["mean"] = float(np.mean(x))
    feat["std"]  = float(np.std(x))
    feat["min"]  = float(np.min(x))
    feat["max"]  = float(np.max(x))
    feat["rms"]  = float(rms(x))
    # robust stats
    try:
        feat["skew"] = float(stats.skew(x))
    except:
        feat["skew"] = 0.0
    try:
        feat["kurtosis"] = float(stats.kurtosis(x))
    except:
        feat["kurtosis"] = 0.0
    feat["spectral_energy"] = float(spectral_energy(x))
    ratios = band_energy_ratios(x, bands=4)
    feat["band0"], feat["band1"], feat["band2"], feat["band3"] = [float(r) for r in ratios]
    return feat
# signal_classification.py
import os
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

def main():
    X_raw, y = generate_signals(n_samples=1000, n_points=200)
    feature_list = [extract_features_from_array(sig) for sig in X_raw]
    feature_names = list(feature_list[0].keys())
    X = pd.DataFrame(feature_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")
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
    for k, v in feats.items():
        print(f"{k:25s}: {v}")

    plot_and_save_fft(first_sig, "fft_before.png", fs=len(first_sig))

    fs = len(first_sig)
    cutoff = 12.0
    filtered = butter_lowpass_filter(first_sig, cutoff=cutoff, fs=fs, order=4)

    # save time-domain before/after
    plt.figure(figsize=(8,3))
    plt.plot(first_sig)
    plt.title("Signal - Before Filtering")
    plt.tight_layout()
    plt.savefig("time_before.png")
    plt.close()
    print("Saved time_before.png")

    plt.figure(figsize=(8,3))
    plt.plot(filtered)
    plt.title("Signal - After Low-pass Filtering")
    plt.tight_layout()
    plt.savefig("time_after.png")
    plt.close()
    print("Saved time_after.png")

    plot_and_save_fft(filtered, "fft_after.png", fs=fs)

    df_filtered = pd.DataFrame({"amplitude_filtered": filtered})
    df_filtered.to_csv("filtered_signal.csv", index=False)
    print("Saved filtered_signal.csv")

    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    fig = plt.figure(figsize=(8,5))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color='skyblue')
    plt.xlabel("Importance"); plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("Saved feature_importance.png")

if __name__ == "__main__":
    main()

# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from feature_utils import extract_features_from_array
import seaborn as sns
import time

st.set_page_config(page_title="Signal Classifier", page_icon="⚡", layout="centered")
st.markdown("""<style>body{background:#0d0f17;color:#fff;}</style>""", unsafe_allow_html=True)
st.title("Signal Classification Dashboard")
st.markdown("Machine Learning + Signal Processing Demo")

def generate_signal(sig_type, freq=5, noise=0.0, n_points=200):
    t = np.linspace(0,1,n_points)
    if sig_type=="sine":
        x = np.sin(2*np.pi*freq*t)
    elif sig_type=="square":
        x = np.sign(np.sin(2*np.pi*freq*t))
    elif sig_type=="sawtooth":
        x = 2*(t - np.floor(t+0.5))
    elif sig_type=="noisy":
        base = np.sin(2*np.pi*freq*t)
        x = base + noise*np.random.randn(len(t))
    else:
        x = np.zeros_like(t)
    return x, t

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq=0.5*fs
    normal_cutoff = cutoff/nyq
    b,a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b,a,data)
    return y

def train_model():
    X=[]; y=[]
    for _ in range(800):
        s = np.random.choice(["sine","square","sawtooth","noisy"])
        f = np.random.randint(3,12)
        n = np.random.uniform(0.0,0.3)
        sig,_ = generate_signal(s,f,n, n_points=200)
        feats = extract_features_from_array(sig)
        X.append(list(feats.values()))
        y.append(s)
    model = RandomForestClassifier()
    model.fit(X,y)
    return model, list(feats.keys())

model, feature_names = train_model()

st.sidebar.header("Controls")
sig_type = st.sidebar.selectbox("Signal type", ["sine","square","sawtooth","noisy"])
freq = st.sidebar.slider("Frequency", 1, 20, 5)
noise = st.sidebar.slider("Noise level", 0.0, 1.0, 0.1)
cutoff = st.sidebar.slider("LPF cutoff (Hz)", 1.0, 50.0, 12.0)
order = st.sidebar.slider("Filter order", 1, 6, 4)
animate = st.sidebar.checkbox("Oscilloscope animation", value=False)
download_wave_png = st.sidebar.checkbox("Enable waveform PNG download", value=True)

sig, t = generate_signal(sig_type, freq, noise, n_points=200)
feats = extract_features_from_array(sig)
pred = model.predict([list(feats.values())])[0]

st.subheader("Time-domain Signal")
fig, ax = plt.subplots()
ax.plot(t, sig, color="cyan")
ax.set_title(f"{sig_type.upper()}  Pred: {pred}")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.subheader("Frequency Spectrum (FFT) - Before")
xf = rfftfreq(len(sig), 1/len(sig))
yf = np.abs(rfft(sig))
fig_fft, axf = plt.subplots()
axf.plot(xf, yf, color="orange")
axf.set_xlabel("Freq (Hz)"); axf.set_ylabel("Amp"); axf.grid(alpha=0.3)
st.pyplot(fig_fft)

# Apply low-pass filter
fs = len(sig)
filtered = butter_lowpass_filter(sig, cutoff=cutoff, fs=fs, order=order)

st.subheader("Time-domain Signal - After Low-pass Filter")
fig2, ax2 = plt.subplots()
ax2.plot(t, filtered, color="lime")
ax2.set_title("Filtered")
ax2.grid(alpha=0.3)
st.pyplot(fig2)

st.subheader("Frequency Spectrum (FFT) - After")
xf2 = rfftfreq(len(filtered), 1/len(filtered))
yf2 = np.abs(rfft(filtered))
fig_fft2, axf2 = plt.subplots()
axf2.plot(xf2, yf2, color="magenta")
axf2.set_xlabel("Freq (Hz)"); axf2.set_ylabel("Amp"); axf2.grid(alpha=0.3)
st.pyplot(fig_fft2)

# Download CSVs
st.subheader("Download data")
csv_original = pd.DataFrame({"time": t, "amplitude": sig}).to_csv(index=False).encode()
csv_filtered = pd.DataFrame({"time": t, "amplitude_filtered": filtered}).to_csv(index=False).encode()
st.download_button("Download original CSV", csv_original, "signal_original.csv", "text/csv")
st.download_button("Download filtered CSV", csv_filtered, "signal_filtered.csv", "text/csv")

# Waveform PNG download
if download_wave_png:
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    st.download_button("Download waveform PNG (before)", img_buf, file_name="wave_before.png", mime="image/png")

# Feature importance chart
st.subheader("Model Feature Importance")
importance = model.feature_importances_
sorted_idx = np.argsort(importance)
fig_fi, ax_fi = plt.subplots(figsize=(7,4))
ax_fi.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color="skyblue")
ax_fi.set_xlabel("Importance"); ax_fi.set_title("Feature Importance"); ax_fi.grid(alpha=0.3)
st.pyplot(fig_fi)

# Multi-signal preview
st.subheader("Multiple signal preview")
variations = [
    {"f":max(1,freq-2),"n":0.0},
    {"f":freq,"n":noise},
    {"f":freq+3,"n":min(1.0,noise+0.1)},
    {"f":max(1,freq-1),"n":min(1.0,noise+0.3)}
]
cols = st.columns(2)
for i,v in enumerate(variations):
    s2, t2 = generate_signal(sig_type, v["f"], v["n"], n_points=100)
    fig2, axx = plt.subplots(figsize=(4,2))
    axx.plot(t2, s2, color="deepskyblue")
    axx.set_title(f"f={v['f']}, noise={round(v['n'],2)}", fontsize=9)
    axx.grid(alpha=0.2)
    with cols[i%2]:
        st.pyplot(fig2)
        plt.close(fig2)

# Upload user CSV
st.subheader("Upload your CSV (time, amplitude)")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        df_user = pd.read_csv(uploaded)
        st.write("Preview:")
        st.line_chart(df_user["amplitude"])
        user_sig = df_user["amplitude"].values
        feats_user = extract_features_from_array(user_sig)
        pred_user = model.predict([list(feats_user.values())])[0]
        st.success(f"Prediction for uploaded signal: {pred_user}")
        st.write(pd.DataFrame([feats_user]))
        filtered_user = butter_lowpass_filter(user_sig, cutoff=cutoff, fs=len(user_sig), order=order)
        st.download_button("Download filtered uploaded CSV", pd.DataFrame({"amplitude_filtered": filtered_user}).to_csv(index=False).encode(), "uploaded_filtered.csv", "text/csv")
    except Exception as e:
        st.error("Invalid CSV. Need columns time, amplitude.")

# Oscilloscope animation (simple)
if animate:
    st.subheader("Oscilloscope (animation)")
    placeholder = st.empty()
    window = 50
    data = np.zeros(window)
    for i in range(200):
        new = np.sin(2*np.pi*(freq + 0.5*np.sin(i/10))* (i/200)) + noise*np.random.randn()
        data = np.roll(data, -1)
        data[-1] = new
        fig_anim, ax_anim = plt.subplots(figsize=(8,2))
        ax_anim.plot(data, color="lime")
        ax_anim.set_ylim(-2, 2)
        ax_anim.axis('off')
        placeholder.pyplot(fig_anim)
        plt.close(fig_anim)
        time.sleep(0.03)

st.markdown("---")
st.markdown("Made with ❤️ for Electronics + ML + DSP")

requirements.txt
numpy
pandas
scikit-learn
matplotlib
scipy
seaborn
streamlit

Thanks for watching out my Repo. Hope you like it and you will fork it....
Add some valuable things if you find ..