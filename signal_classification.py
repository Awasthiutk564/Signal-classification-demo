# -----------------------------
# Signal Classification using ML (With Feature Extraction + Sawtooth)
# -----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🔹 Import feature extractor
from feature_utils import extract_features_from_array


# Step 1: Generate signals (Sine + Square + Sawtooth)
def generate_signals(n_samples=1000, n_points=100):
    X = []
    y = []
    for _ in range(n_samples):
        signal_type = np.random.choice(['sine', 'square', 'sawtooth'])  # 🔥 NEW: sawtooth added
        t = np.linspace(0, 1, n_points)

        if signal_type == 'sine':
            signal = np.sin(2 * np.pi * 5 * t)

        elif signal_type == 'square':
            signal = np.sign(np.sin(2 * np.pi * 5 * t))

        elif signal_type == 'sawtooth':   # 🔥 NEW BLOCK
            signal = 2 * (t - np.floor(t + 0.5))

        X.append(signal)
        y.append(signal_type)

    return np.array(X), np.array(y)


# Step 2: Prepare data
X_raw, y = generate_signals()

feature_list = []
for sig in X_raw:
    feats = extract_features_from_array(sig)
    feature_list.append(feats)

X = pd.DataFrame(feature_list)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Test
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy (with Sine + Square + Sawtooth): {acc*100:.2f}%\n")

# Step 5: Show graph
index = np.random.randint(0, len(X_test))
actual_index = list(X_test.index)[index]
actual_signal = X_raw[actual_index]

plt.plot(actual_signal)
plt.title(f"Actual: {y_test[index]} | Predicted: {y_pred[index]}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Step 6: Print extracted features
print("\n📌 Extracted Features for this test signal:")
feats = extract_features_from_array(actual_signal)
for k, v in feats.items():
    print(f"{k:25s}: {v}")
