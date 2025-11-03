# -----------------------------
# Signal Classification using ML
# -----------------------------
# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2: Generate sample signals
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
        else:
            signal = np.random.randn(n_points)  # noisy signal

        X.append(signal)
        y.append(signal_type)
    return np.array(X), np.array(y)

# Step 3: Prepare data
X, y = generate_signals()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train ML model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Test model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc*100:.2f}%")

# Step 6: Visualize one random signal and prediction
index = np.random.randint(0, len(X_test))
plt.plot(X_test[index])
plt.title(f"Actual: {y_test[index]} | Predicted: {y_pred[index]}")
plt.show()
