# src/data/preprocess.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple

def bandpass_filter(sig: np.ndarray, fs: int, low: float = 4.0, high: float = 45.0, order: int = 4) -> np.ndarray:
    ny = 0.5 * fs
    b, a = butter(order, [low/ny, high/ny], btype='band')
    return filtfilt(b, a, sig, axis=0)

def baseline_correct(stim: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    mu = baseline.mean(axis=0, keepdims=True)
    return stim - mu

def window_signal(sig: np.ndarray, fs: int, win_sec: float = 2.0, overlap: float = 0.5) -> np.ndarray:
    # sig: (T, C) -> (N, W, C)
    win = int(win_sec * fs)
    hop = max(1, int(win * (1 - overlap)))
    T, C = sig.shape
    out = []
    for start in range(0, max(T - win + 1, 0), hop):
        out.append(sig[start:start+win])
    return np.stack(out, axis=0) if out else np.empty((0, win, C), dtype=sig.dtype)
