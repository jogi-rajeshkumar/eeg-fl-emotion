# Federated Learning for EEG-Based Emotion Recognition

This repository documents the implementation and results of an MSc Artificial Intelligence dissertation project titled: **"Federated Learning for EEG-Based Emotion Recognition Across Edge Devices"**. The core objective of this research is to rigorously evaluate the performance and critical trade-offs between conventional centralized deep learning and a privacy-preserving approach utilizing Federated Learning (FL). The entire study is anchored on the publicly available, multi-modal **DREAMER dataset**.

## 📌 Project Overview: The Privacy Imperative

The fundamental motivation for this research is to address the significant challenge of managing sensitive biomedical data, specifically electroencephalogram (EEG) signals, which traditionally requires centralized storage. Moving away from this paradigm, we implement a **1D Convolutional Neural Network (CNN-1D)** architecture, specifically optimized for edge deployment, to perform a binary classification task. This classification focuses on two primary emotional dimensions derived from the EEG signals: **Arousal** (high vs. low) and **Valence** (positive vs. negative) states.

## Key Features and Design Principles

The architecture and implementation were guided by principles of privacy, realism, and empirical measurement:

* **Privacy-First Implementation:** The entire Federated Learning infrastructure is built using the industry-leading **Flower (FLWR)** framework. Crucially, the raw, sensitive EEG data is guaranteed never to leave the secure confines of the client device, preserving subject privacy throughout the training process.
* **Subject-Wise Partitioning (Non-IID Data):** To accurately model real-world deployment scenarios—such as a remote patient monitoring system where each client is a different person—the DREAMER dataset is partitioned such that each participant is treated as a unique federated client. This design naturally results in a highly non-Independent and Identically Distributed (**non-IID**) data setting, which is a major challenge in FL.
* **Comprehensive Resource Monitoring:** To provide a complete picture of the FL system's viability for resource-constrained environments (like the Internet of Medical Things, or IoMT), the project includes built-in, granular logging for critical performance metrics:
    * **Communication Overhead:** Detailed measurement of data transfer (uplink and downlink bytes) per training round.
    * **Training Latency:** Monitoring of the time required for local model training and aggregation.

---

## 🏗 System Architecture and Methodology

The complete system pipeline is segmented into data handling and model specification.

### 1. Preprocessing Pipeline: Ensuring Signal Fidelity

The raw EEG signals from the DREAMER dataset undergo a rigorous, multi-stage cleaning and transformation process to enhance signal quality and prepare the data for the CNN-1D:



* **Band-pass Filtering:** A 4th-order Butterworth filter is applied with a passband of **4-45 Hz**. This is a standard procedure to isolate physiologically relevant rhythms while suppressing DC drift and high-frequency noise.
* **Baseline Correction:** Inter-subject variability is reduced by subtracting the mean activity recorded during the pre-stimulus baseline period from the activity recorded during the emotional stimulus period.
* **Normalization:** Per-subject **Z-score standardization** is performed This step is critical in FL, ensuring that the local data distribution of each client is normalized, preventing a single subject's signals from unduly influencing the global model update.
* **Segmentation:** Continuous EEG streams are broken down into **2.0-second time windows** with a **50% overlap**. This method increases the effective size of the dataset and provides a robust feature representation.

### 2. Model Architecture (CNN-1D): Optimization for Edge Devices

An intentionally lightweight CNN-1D model was designed to be efficient for deployment on resource-constrained edge devices, featuring approximately 0.5 million trainable parameters:

* **Conv1D Layer 1:** 32 filters, kernel size of 7, followed by Batch Normalization, a ReLU activation function, and a Max Pooling layer.
* **Conv1D Layer 2:** 64 filters, kernel size of 5, followed by Batch Normalization, a ReLU activation function, and a Max Pooling layer.
* **Dense Head:** A final fully connected layer of 64 units, incorporating a **Dropout layer (p=0.3)** for regularization, before the final binary classification output.

---

## 📊 Results and Analysis: Evaluating the Trade-offs

The experimental phase focused on a head-to-head comparison of performance metrics between the traditional centralized approach and the privacy-preserving Federated Learning deployment.

### Centralized vs. Federated Performance Comparison

The results demonstrate the feasibility of FL for this task, though they also expose known challenges associated with non-IID data distributions.

| Task | Setting | Accuracy | F1-Score | ROC-AUC | Key Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arousal** | Centralized | 76.9% | 0.84 | 0.81 | High performance benchmark. |
| **Arousal** | Federated | 71.1% | 0.83 | 0.53 | F1-Score maintains parity, but ROC-AUC drop highlights difficulty in classifying all subjects equally. |
| **Valence** | Centralized | 71.8% | 0.79 | 0.78 | Consistent performance with Arousal. |
| **Valence** | Federated | 71.1% | 0.83 | 0.53 | Similar trend: High F1-Score suggests reliable prediction, but low ROC-AUC indicates poor generalization across non-IID clients. |

**Summary:** The Federated Learning model achieved Accuracy and F1-scores comparable to the centralized model. However, the significantly lower **ROC-AUC** for the federated approach highlights a key challenge: the non-IID nature of subject-wise data partitioning makes it difficult for the aggregated global model to perform uniformly well across all individual clients.

### Communication Overhead: The Critical Factor for IoMT

A vital contribution of this project is the empirical measurement of communication costs, a key barrier to widespread FL adoption:

* **Payload Size:** For a setup involving 16 federated clients, the total data transfer per round (model weights and gradients) was measured to be approximately **~3.8 MB**.
* **Scalability and Baseline:** This figure establishes a crucial, empirically-validated baseline for data bandwidth requirements in bandwidth-constrained IoMT environments. This metric is essential for future efforts focused on optimizing FL communication strategies.

---

## 📂 Repository Structure

The project is logically organized to facilitate reproducibility and future extensions:

```text
eeg-fl-emotion/
├── Dataset/             # Placeholder for the DREAMER.mat file (External access required)
├── env/                 # Includes requirements.txt for environment management 
├── scripts/             # Execution scripts (run_central.sh & run_fed.sh) 
├── results/             # Storage for generated output: CSV logs, ROC plots, and models 
└── src/                 # Source code directory 
    ├── data/            # Data preprocessing pipeline and DreamerWindows class 
    ├── fed/             # Flower server, client logic, and orchestration 
    ├── models/          # PyTorch implementation of the CNN-1D architecture 
    └── train/           # Utilities for the centralized training benchmark 