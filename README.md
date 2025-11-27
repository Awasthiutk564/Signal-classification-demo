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
