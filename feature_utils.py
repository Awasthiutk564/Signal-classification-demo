# feature_utils.py
import numpy as np
from scipy.fft import rfft

def extract_features_from_array(x: np.ndarray) -> dict:
    """
    x: 1D signal (numpy array)
    returns: dict of simple time + freq domain features
    """
    feats = {}
    # Time-domain
    feats["mean"] = float(np.mean(x))
    feats["std"] = float(np.std(x))
    feats["min"] = float(np.min(x))
    feats["max"] = float(np.max(x))
    feats["rms"] = float(np.sqrt(np.mean(x**2)))

    # Frequency-domain (magnitude spectrum)
    X = np.abs(rfft(x))
    total = float(np.sum(X) + 1e-9)
    feats["fft_energy"] = total

    # 4 equal bands (relative energy)
    n = len(X)
    for i in range(4):
        s = int(i * n/4)
        e = int((i+1) * n/4)
        feats[f"band_{i}_energy_ratio"] = float(np.sum(X[s:e]) / total)
    return feats
