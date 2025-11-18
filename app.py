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
st.title("⚡ Signal Classification Dashboard")
st.markdown("Machine Learning + Signal Processing + Live Demo")

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
    for _ in range(900):
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
        # offer download of filtered uploaded signal
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
        # shift and append new sample (simulate)
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
