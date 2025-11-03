import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📶 Signal Classification Web App")
st.write("Upload ya generate a signal aur dekho prediction + graph!")

# ----- Generate Signal -----
signal_type = st.selectbox("Signal Type Choose Karo:", ["Sine", "Square"])
freq = st.slider("Frequency (Hz):", 1, 10, 5)
samples = 100

t = np.linspace(0, 1, samples)
if signal_type == "Sine":
    signal = np.sin(2 * np.pi * freq * t)
else:
    signal = np.sign(np.sin(2 * np.pi * freq * t))

# ----- Plot Signal -----
st.subheader("Generated Signal:")
fig, ax = plt.subplots()
ax.plot(t, signal)
st.pyplot(fig)

# ----- Prepare Data -----
X = np.array([np.sin(2 * np.pi * f * t) for f in range(1, 11)] +
             [np.sign(np.sin(2 * np.pi * f * t)) for f in range(1, 11)])
y = np.array([0]*10 + [1]*10)  # 0 = Sine, 1 = Square

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# ----- Prediction -----
input_signal = signal.reshape(1, -1)
pred = model.predict(input_signal)[0]

st.subheader("🔍 Prediction Result:")
if pred == 0:
    st.success("Predicted: Sine Wave ✅")
else:
    st.success("Predicted: Square Wave ✅")

st.write(f"Model Accuracy: **{acc*100:.2f}%**")
