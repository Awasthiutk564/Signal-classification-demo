import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from feature_utils import extract_features_from_array
from scipy.fft import rfft, rfftfreq
import seaborn as sns

# -----------------------------------------------------------
# STREAMLIT SETTINGS
# -----------------------------------------------------------
st.set_page_config(
    page_title="Signal Classifier",
    page_icon="⚡",
    layout="centered"
)

st.markdown("""
<style>
body { background-color: #0d0f17; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Signal Classification Dashboard")
st.markdown("### Machine Learning + Signal Processing + Electronics")


# -----------------------------------------------------------
# SIGNAL GENERATION
# -----------------------------------------------------------
def generate_signal(sig_type, freq=5, noise=0.0):
    t = np.linspace(0, 1, 200)

    if sig_type == "sine":
        x = np.sin(2 * np.pi * freq * t)

    elif sig_type == "square":
        x = np.sign(np.sin(2 * np.pi * freq * t))

    elif sig_type == "sawtooth":
        x = 2 * (t - np.floor(t + 0.5))

    elif sig_type == "noisy":
        base = np.sin(2 * np.pi * freq * t)
        x = base + noise * np.random.randn(len(t))

    return x, t


# -----------------------------------------------------------
# MODEL TRAINING (INTERNAL)
# -----------------------------------------------------------
def train_model():
    X = []
    y = []

    for _ in range(900):
        sig_type = np.random.choice(["sine", "square", "sawtooth", "noisy"])
        freq = np.random.randint(3, 12)
        noise = np.random.uniform(0.0, 0.3)

        sig, _ = generate_signal(sig_type, freq, noise)
        feats = extract_features_from_array(sig)

        X.append(list(feats.values()))
        y.append(sig_type)

    model = RandomForestClassifier()
    model.fit(X, y)
    return model, list(feats.keys())


model, feature_names = train_model()


# -----------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------
st.sidebar.header("⚙ Signal Controls")

sig_type = st.sidebar.selectbox("Choose Signal Type:",
    ["sine", "square", "sawtooth", "noisy"])

freq = st.sidebar.slider("Frequency", 1, 20, 5)
noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)


# -----------------------------------------------------------
# GENERATE + PREDICT
# -----------------------------------------------------------
sig, t = generate_signal(sig_type, freq, noise)
feats = extract_features_from_array(sig)
feat_array = np.array([list(feats.values())])

pred = model.predict(feat_array)[0]


# -----------------------------------------------------------
# MAIN SIGNAL PLOT
# -----------------------------------------------------------
st.subheader("📈 Time-Domain Signal")
fig, ax = plt.subplots()
ax.plot(t, sig, color="cyan")
ax.set_title(f"{sig_type.upper()} | Predicted: {pred}")
ax.grid(True, alpha=0.3)
st.pyplot(fig)


# -----------------------------------------------------------
# FFT FREQUENCY SPECTRUM
# -----------------------------------------------------------
st.subheader("🎧 Frequency Spectrum (FFT)")

xf = rfftfreq(len(sig), 1/len(sig))
yf = np.abs(rfft(sig))

fig_fft, ax_fft = plt.subplots()
ax_fft.plot(xf, yf, color="orange")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Amplitude")
ax_fft.grid(alpha=0.3)
st.pyplot(fig_fft)


# -----------------------------------------------------------
# FEATURES
# -----------------------------------------------------------
st.subheader("🧮 Extracted Features")
st.write(pd.DataFrame([feats]))


# -----------------------------------------------------------
# DOWNLOAD CSV
# -----------------------------------------------------------
csv_data = pd.DataFrame({"time": t, "amplitude": sig})
csv_bytes = csv_data.to_csv(index=False).encode()

st.subheader("📥 Download Generated Signal")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name="generated_signal.csv",
    mime="text/csv"
)


# -----------------------------------------------------------
# MULTI-SIGNAL GRID
# -----------------------------------------------------------
st.subheader("🔎 Multi-Signal Preview")

variations = [
    {"label": f"{sig_type} (freq {max(1,freq-2)})", "freq": max(1, freq-2), "noise": 0},
    {"label": f"{sig_type} (freq {freq})", "freq": freq, "noise": noise},
    {"label": f"{sig_type} (freq {freq+3})", "freq": freq+3, "noise": noise},
    {"label": f"{sig_type} (noisy)", "freq": freq, "noise": min(1.0, noise+0.3)}
]

cols = st.columns(2)
for i, v in enumerate(variations):
    s2, tt2 = generate_signal(sig_type, v["freq"], v["noise"])
    fig2, ax2 = plt.subplots(figsize=(4,2))
    ax2.plot(tt2, s2, color="deepskyblue")
    ax2.set_title(v["label"], fontsize=9)
    ax2.grid(alpha=0.3)
    with cols[i % 2]:
        st.pyplot(fig2)
        plt.close(fig2)


# -----------------------------------------------------------
# UPLOAD USER SIGNAL
# -----------------------------------------------------------
st.subheader("📤 Upload Your Own CSV Signal")

uploaded = st.file_uploader("Upload CSV file with columns: time, amplitude")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)

        st.write("Uploaded Signal Preview:")
        st.line_chart(df["amplitude"])

        user_signal = df["amplitude"].values
        feats_u = extract_features_from_array(user_signal)
        pred_u = model.predict([list(feats_u.values())])[0]

        st.success(f"Prediction for uploaded signal: **{pred_u}**")
        st.write(pd.DataFrame([feats_u]))

    except:
        st.error("Invalid CSV format! Must contain 'time' and 'amplitude' columns.")


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.markdown("Made with ❤️ for Electronics + Machine Learning + DSP")
