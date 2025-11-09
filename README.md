# 📶 Signal Classification using Machine Learning  
This project classifies signals like **Sine** and **Square** waves using **Machine Learning**,  
and now includes a **Feature Extraction module (Time + Frequency domain)** for smarter predictions.

---

## 🧠 What This Project Does
- Generates synthetic signals (sine, square)
- Extracts meaningful features:
  - Mean, Standard Deviation  
  - Min, Max  
  - RMS (Root Mean Square)  
  - FFT Energy  
  - Frequency Energy Bands  
- Trains a **Random Forest Classifier**
- Shows model accuracy + graph + extracted feature values

---

## 📂 Project Files
| File | Description |
|------|-------------|
| `signal_classification.py` | Main ML script (training + testing + plotting + features) |
| `feature_utils.py` | Time & frequency domain feature extraction |
| `app.py` | (Optional) Streamlit web app for UI |
| `signal_app_screenshot.png` | Output screenshot |
| `README.md` | Documentation |

---

## 🚀 How to Run
1. Install required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib scipy
