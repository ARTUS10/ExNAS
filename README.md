# ExNAS – Reproducibility Archive

This repository reproduces all experiments reported in the paper:
**"Experiential Neural Architecture Selection (ExNAS): Inference-time neuron-level adaptation with experiential memory"**.

## Contents
- `exnas_cnn.py` — CNN experimental pipeline (CIFAR-10, CPU).
- `exnas_qwen_v2.py` — Qwen2-1.5B static + SeqGate v2 evaluation.
- `exnas_seqgate_v2_distilgpt2.py` — DistilGPT-2 static + SeqGate v2 evaluation.
- `requirements.txt` — exact library versions used.

## Data
- **CNNs**: CIFAR-10 via `torchvision`; if unavailable, the script falls back to a synthetic balanced dataset (6k train / 1k test).
- **Transformers (LM)**: WikiText-2 (streaming) via `datasets`.

## Hardware & Scope
- CNNs: CPU only.
- Transformers: NVIDIA T4 with FP16/BF16 unless stated otherwise.

## Tested Requirements
(see `requirements.txt`)

- python 3.12  
- torch 2.3.1 (CUDA 12.x)  
- torchvision 0.18.1  
- transformers 4.41.1  
- datasets 2.20.0  
- numpy 2.1.3  
- scipy 1.13.1  
- tqdm 4.66.4  

## How to reproduce

### 1. Install dependencies
pip install -r requirements.txt

### 2. CNN (CPU) experiments
python exnas_cnn.py

### 3. Transformers (LM)
### Qwen2-1.5B (static + SeqGate v2)
python exnas_qwen_v2.py

#### DistilGPT-2 (SeqGate v2)
python exnas_seqgate_v2_distilgpt2.py

### Notes
- No internet is required once datasets are cached.
- Scripts automatically log hyperparameters, seeds, and full system configuration.

# ExNAS
Experiential Neural Architecture Selection for real-time inference optimization
