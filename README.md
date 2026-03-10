# Time-Series Anomaly Detection: Foundation Models vs. Deep Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)]()
[![uv](https://img.shields.io/badge/uv-Fast%20Package%20Manager-purple)]()

## Overview
This repository explores the architectural trade-offs between custom deep learning and zero-shot foundation models for industrial time-series anomaly detection. 

Using the **Server Machine Dataset (SMD)**, this project benchmarks two distinct approaches to identifying infrastructure failures and cyber anomalies:
1. **The Baseline (LSTM Autoencoder):** A custom PyTorch model trained from scratch to learn the mathematical "shape" of healthy server data, flagging anomalies via reconstruction error spikes.
2. **The SOTA (Amazon Chronos):** Evaluating Amazon's `chronos-t5-small` foundation model out-of-the-box (zero-shot inference), utilizing tokenized time-series forecasting and probabilistic bounds.



## Repository Structure

```text
.
├── src/
│   ├── preprocess_smd.py      # Data ingestion & sliding window tensor generation
│   ├── lstm_autoencoder.py    # PyTorch Autoencoder architecture
│   ├── train.py               # Training loop and Adam optimizer logic
│   ├── main.py                # LSTM execution and evaluation pipeline
│   └── chronos_inference.py   # Zero-shot evaluation using Amazon Chronos
├── data/                      # (Git-ignored) Raw SMD text files
├── .gitignore
└── README.md
```

## Setup & Installation

### 1. Clone the repository and install dependencies
```bash
git clone https://github.com/mattcarp12/time-series-anomaly-detection
cd time-series-anomaly-detection
uv venv
uv pip install -r pyproject.toml 
```

### 2. Download the Server Machine Dataset (SMD):
```bash
mkdir -p data/smd
cd data/smd
git clone --depth 1 --filter=blob:none --sparse https://github.com/NetManAIOps/OmniAnomaly.git .
git sparse-checkout set ServerMachineDataset
cd ../..
```

## Running the Pipelines

### Run the PyTorch LSTM Baseline:
```bash
uv run src/main.py
```

### Run the Chronos Zero-Shot Inference:
```bash
uv run src/chronos_inference.py
```
