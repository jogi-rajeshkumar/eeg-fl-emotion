# Federated Learning for EEG-Based Emotion Recognition Across Edge Devices

[cite_start]This repository contains the official implementation of the MSc Artificial Intelligence dissertation project: **"Federated Learning for EEG-Based Emotion Recognition Across Edge Devices"**[cite: 11]. [cite_start]The project investigates the feasibility of using Federated Learning (FL) as a privacy-preserving alternative to centralized machine learning for processing sensitive biomedical (EEG) data[cite: 21, 28].

## Project Overview
[cite_start]Traditional centralized machine learning requires pooling sensitive EEG data into a single repository, raising significant privacy and security concerns[cite: 20, 105]. [cite_start]This project implements a **1D Convolutional Neural Network (CNN-1D)** within the **Flower (FLWR)** framework to recognize emotions (Arousal and Valence) using the **DREAMER dataset**[cite: 21, 24, 416].

### Key Features
* [cite_start]**Centralized Baseline:** Compact and expressive CNN-1D models trained on pooled data[cite: 414].
* [cite_start]**Federated Implementation:** Subject-wise partitioning where one subject corresponds to one client[cite: 416].
* [cite_start]**Communication Analysis:** Systematic logging of data overhead (uplink/downlink) per training round[cite: 418, 533].
* [cite_start]**Privacy-by-Design:** Raw EEG data remains local; only model updates are exchanged with the coordinating server[cite: 110, 580].

---

## System Architecture



### Preprocessing Pipeline
[cite_start]To handle noisy EEG signals, a literature-compliant pipeline was implemented[cite: 438, 1514]:
1. [cite_start]**Band-pass Filtering:** 4–45 Hz using a 4th-order Butterworth filter to isolate affect-relevant rhythms[cite: 439, 440].
2. [cite_start]**Baseline Correction:** Subtracting the mean of the resting-state segment from the stimulus segment[cite: 442, 638].
3. [cite_start]**Z-Score Normalization:** Per-subject standardization to improve optimization stability[cite: 444, 641].
4. [cite_start]**Windowing:** 2.0s segments (256 samples) with 50% overlap[cite: 446, 644].

### Model Architecture (CNN-1D)
[cite_start]The architecture is designed for efficiency on resource-constrained devices[cite: 478, 692]:
* [cite_start]**Input:** [batch, 14 channels, 256 time steps][cite: 468].
* [cite_start]**Conv Blocks:** Two layers with kernels (7 and 5) followed by BatchNorm, ReLU, and MaxPool[cite: 470, 471].
* [cite_start]**Head:** Flatten -> Linear (64 units) -> ReLU -> Dropout (0.3) -> Linear output[cite: 471].

---

## Experimental Results

### Performance Comparison
[cite_start]The study compared a Centralized baseline against the Federated (FedAvg) approach[cite: 1727, 1751]:

| Metric | Arousal (Central) | Arousal (Fed) | Valence (Central) | Valence (Fed) |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 76.98% | 71.09% | 71.87% | 71.09% |
| **F1-Score** | 0.8468 | 0.8311 | 0.7926 | 0.8311 |
| **ROC-AUC** | 0.8155 | 0.5352 | 0.7848 | 0.5297 |



### Communication Overhead
[cite_start]Practical feasibility was assessed by recording the data exchanged[cite: 1409]. [cite_start]For each round, the payload size was recorded[cite: 536, 1410]:
* [cite_start]**Uplink/Downlink per Client:** ~971,808 bytes[cite: 1419, 1420].
* [cite_start]**Total per Round (16 Clients):** ~3.8 MB[cite: 1730, 1857].

---

## Repository Structure
[cite_start]The project is organized for high reproducibility[cite: 564, 1455]:
```text
eeg-fl-emotion/
├── Dataset/             # DREAMER.mat (requires academic access)
├── env/                 # requirements.txt for environment setup
├── results/             # CSVs, PNGs, and Model Checkpoints
│   ├── central/         # Baseline metrics (metrics.csv, roc.png)
│   └── fed/             # FL metrics and comm_log.csv
├── scripts/             # run_central.sh, run_fed.sh
└── src/                 
    ├── data/            # Preprocessing and dataset classes
    ├── fed/             # Flower Server, Client, and Launcher
    ├── models/          # CNN-1D architecture
    ├── train/           # Centralized training utilities
    └── eval/            # Comparative evaluation scripts
