# ⚡ Signal Classification using Machine Learning + DSP + Streamlit

A complete end-to-end project that classifies signals  
(**Sine, Square, Sawtooth, Noisy**) using Machine Learning and Signal Processing.  
It also includes a fully interactive **Streamlit Web App**.

---

## 🚀 Features

### ✔ Machine Learning  
- Random Forest Classifier  
- Feature extraction (mean, std, kurtosis, skewness, peaks, freq-domain features)  
- Accuracy score  
- Confusion matrix heatmap  
- Classification report  
- Feature importance (Explainability)  
- Saves graphs as PNG (FFT, Confusion Matrix, Feature Importance)

### ✔ Signal Processing (DSP)
- Time-domain signal generation  
- Frequency spectrum using FFT  
- Noisy signal simulation  
- Multiple signal previews  
- Ability to generate and analyze custom signals  

### ✔ Streamlit Web App
- Live signal generation using sliders  
- Real-time ML prediction  
- FFT visualization  
- Feature importance visualization  
- Multi-signal comparison grid  
- Upload your own CSV signal and get prediction  
- Download generated signal as CSV  

---

## 📂 Project Structure

signal_classification_project/
│
├── app.py # Streamlit Dashboard
├── signal_classification.py # ML + DSP training/plots
├── feature_utils.py # Feature extraction functions
│
├── data/
│ └── signals_sample.csv # Generated sample dataset
│
├── confusion_matrix.png
├── fft_spectrum.png
├── feature_importance.png
│
└── README.md

---

## ▶️ Run ML Script

python signal_classification.py

This will:
- Train the model  
- Display confusion matrix  
- Show multi-signal plots  
- Plot FFT  
- Show feature importance  
- Save CSV + PNG files  

---

## 🌐 Run Streamlit Web App

python -m streamlit run app.py

Then open:

http://localhost:8501/

---

## 📤 Upload Format (For Custom Signal Prediction)

Your CSV must look like:

time,amplitude
0.00,0.12
0.01,0.14
0.02,0.20

---

## 💡 Why This Project Is Hackathon-Ready
- ML + DSP + Web App combined  
- Electronics + CS crossover  
- Professional visualizations  
- Clean code  
- End-to-end pipeline  
- Great for resume, GitHub, and demo  

---

## 👤 Author  
**Utkarsh Awasthi**

---

## ⭐ If you like this project, star the repository!
