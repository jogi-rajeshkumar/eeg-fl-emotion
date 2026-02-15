# src/data/load_dreamer.py
import warnings
warnings.filterwarnings("ignore")

import scipy.io as sio
import numpy as np
from typing import Dict, List, Tuple

def binarise(x: int, thr: int = 3) -> int:
    return 1 if int(x) >= thr else 0

def load_dreamer_mat(path: str) -> Tuple[Dict[str, List[dict]], dict]:
    """
    Returns:
      subjects: { 'subj_01': [ { 'eeg_baseline': (Tb,14), 'eeg_stim': (Ts,14), 'val':int, 'aro':int }, ... ], ... }
      meta: { 'eeg_fs': 128, 'ecg_fs': 256, 'electrodes': [...14...] }
    """
    M = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    D = M['DREAMER']
    subjects = {}
    for si, S in enumerate(D.Data, start=1):
        trials = []
        ntr = len(S.EEG.stimuli)
        for t in range(ntr):
            baseline = np.asarray(S.EEG.baseline[t], dtype=np.float32)   # (Tb,14)
            stim     = np.asarray(S.EEG.stimuli[t], dtype=np.float32)    # (Ts,14)
            trials.append({
                "eeg_baseline": baseline,
                "eeg_stim": stim,
                "val": binarise(int(S.ScoreValence[t])),
                "aro": binarise(int(S.ScoreArousal[t])),
            })
        subjects[f"subj_{si:02d}"] = trials
    meta = {
        "eeg_fs": int(D.EEG_SamplingRate),
        "ecg_fs": int(D.ECG_SamplingRate),
        "electrodes": list(D.EEG_Electrodes),
    }
    return subjects, meta
