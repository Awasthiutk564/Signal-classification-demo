# 🎵 Signal Classification using Machine Learning

This project classifies different types of signals such as **sine wave**, **square wave**, **sawtooth wave**, etc., using a simple Machine Learning model.

## 📂 Files Overview
- `signal_classification.py` → ML model training and accuracy
- `app.py` → Optional Streamlit web app (for visualization)
- `README.md` → Project description

## ⚙️ How to Run
1. Open terminal in the project folder  
2. Run:
   ```bash
   python signal_classification.py
## 🧮 Feature Extraction
For each signal, I plan to add a feature extraction step that calculates:
- Mean, Standard Deviation, Minimum, Maximum
- RMS (Root Mean Square)
- Frequency-domain energy using FFT

These features can make the ML model smarter and more generalizable.  
Next goal: Implement feature extraction in Python and retrain the model with these values.
