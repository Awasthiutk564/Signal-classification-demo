import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_utils import extract_features_from_array
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# -----------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# -----------------------------------------------------------
st.set_page_config(
    page_title="Signal Classifier",
    page_icon="⚡",
    layout="centered"
)

# Dark Theme
st.markdown("""
<style>
body { background-color: #0d0f17; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Signal Classification Dashboard")
st.markdown("### Machine Learning + Electronics Demo")

# -----------------------------------------------------------
# SIGNAL GENERATOR
# -----------------------------------------------------------
def generate_signal(sig_type, freq=5, noise=0.0):
    t = np.linspace(0, 1, 100)

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
# TRAIN INTERNAL MODEL (SMALL RF MODEL)
# -----------------------------------------------------------
def train_model():
    X = []
    y = []

    for _ in range(800):
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

sig_type = st.sidebar.selectbox("Choose signal type:", 
    ["sine", "square", "sawtooth", "noisy"]
)

freq = st.sidebar.slider("Frequency", 1, 20, 5)
noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)


# -----------------------------------------------------------
# GENERATE SIGNAL + PREDICT
# -----------------------------------------------------------
sig, t = generate_signal(sig_type, freq, noise)
feats = extract_features_from_array(sig)
feat_array = np.array([list(feats.values())])

pred = model.predict(feat_array)[0]

# -----------------------------------------------------------
# PLOT SIGNAL
# -----------------------------------------------------------
st.subheader("📈 Signal Preview")
fig, ax = plt.subplots()
ax.plot(t, sig, color="cyan")
ax.set_title(f"Actual: {sig_type}  |  Predicted: {pred}")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# -----------------------------------------------------------
# FEATURES TABLE
# -----------------------------------------------------------
st.subheader("🧮 Extracted Features")
st.write(pd.DataFrame([feats]))

# -----------------------------------------------------------
# PREDICTION RESULT
# -----------------------------------------------------------
st.subheader("🎯 Prediction")
st.success(f"**Predicted Signal Type: {pred}**")
# -----------------------------------------------------------
# DOWNLOAD SIGNAL CSV
# -----------------------------------------------------------
csv_data = pd.DataFrame({
    "time": t,
    "amplitude": sig
})

csv_bytes = csv_data.to_csv(index=False).encode()

st.subheader("📥 Download Generated Signal")
st.download_button(
    label="Download Signal CSV",
    data=csv_bytes,
    file_name="generated_signal.csv",
    mime="text/csv"
)
# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.markdown("Made with ❤️ for Electronics + Machine Learning")
