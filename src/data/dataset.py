# src/data/dataset.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from .load_dreamer import load_dreamer_mat
from .preprocess import bandpass_filter, baseline_correct, window_signal

class DreamerWindows(Dataset):
    def __init__(self, mat_path: str, target: str = "arousal", fs: int = 128,
                 win_sec: float = 2.0, overlap: float = 0.5, subjects_subset: List[str] = None):
        assert target in ("arousal", "valence")
        subjects, meta = load_dreamer_mat(mat_path)
        if subjects_subset:
            subjects = {k: subjects[k] for k in subjects_subset}
        self.fs = fs if fs is not None else int(meta["eeg_fs"])

        X, y = [], []
        for subj, trials in subjects.items():
            for tr in trials:
                stim = bandpass_filter(baseline_correct(tr["eeg_stim"], tr["eeg_baseline"]), self.fs)
                wins = window_signal(stim, fs=self.fs, win_sec=win_sec, overlap=overlap)  # (N, W, C)
                if wins.shape[0] == 0:
                    continue
                label = tr["aro"] if target == "arousal" else tr["val"]
                y.extend([label] * wins.shape[0])
                X.append(wins)

        if X:
            X = np.concatenate(X, axis=0)  # (N, W, C)
        else:
            X = np.empty((0, int(win_sec*self.fs), 14), dtype=np.float32)

        self.X = torch.from_numpy(X).permute(0, 2, 1).contiguous().float()  # (N, C, W)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
