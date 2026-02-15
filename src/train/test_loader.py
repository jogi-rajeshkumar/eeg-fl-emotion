import warnings
warnings.filterwarnings("ignore")

import os
from src.data.load_dreamer import load_dreamer_mat

def main():
    path = "/home/rajeshkumarjogi/Desktop/eeg-fl-emotion/Dataset/DREAMER.mat"
    assert os.path.exists(path), f"File not found: {path}"

    subjects, meta = load_dreamer_mat(path)

    print("=== Meta Info ===")
    print("EEG fs:", meta["eeg_fs"], "Hz")
    print("ECG fs:", meta["ecg_fs"], "Hz")
    print("Electrodes:", meta["electrodes"])

    # show first subject
    subj_id = list(subjects.keys())[0]
    print("\nFirst subject:", subj_id)
    first_trial = subjects[subj_id][0]
    print("Baseline shape:", first_trial["eeg_baseline"].shape)
    print("Stimulus shape:", first_trial["eeg_stim"].shape)
    print("Valence label:", first_trial["val"], "Arousal label:", first_trial["aro"])

    # count trials
    total_trials = sum(len(trials) for trials in subjects.values())
    print("\nTotal subjects:", len(subjects))
    print("Total trials:", total_trials)

if __name__ == "__main__":
    main()
