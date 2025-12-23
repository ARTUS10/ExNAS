# ExNAS: Experiential Neural Architecture Selection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org/)

## Title

**ExNAS: Experiential Neural Architecture Selection for Real-Time Inference Optimization**

## Description

ExNAS is a system that performs real-time, fine-grained substructure selection (channels, neurons, or heads, depending on architecture) during inference by leveraging a lightweight experiential memory. Unlike traditional approaches that treat each input independently, ExNAS records layer-wise activation fingerprints, retrieves similar past contexts, and applies structurally guided selection across non-consecutive layers.

### Key Features

- **Experience-guided selection**: Records process signatures from guardrail-satisfying runs, not raw content
- **Transversal gating**: Selects substructures across non-consecutive layers within the same forward pass
- **No retraining required**: All adaptations are post-hoc and weight-frozen
- **Budget-constrained**: Explicit per-layer and global budgets with quality guardrails (ΔPPL ≤ 1%)
- **Kernel-aware**: Respects hardware alignment (×64/×128) for optimized inference

### Results Summary

| Architecture | Configuration | Speedup | Quality Impact |
|-------------|---------------|---------|----------------|
| SmallCNN (CPU) | Fast | +3.8% throughput | Accuracy trade-off |
| SmallCNN (CPU) | Grid-best | +8.5% throughput | Accuracy trade-off |
| Qwen2-1.5B (T4 GPU) | SeqGate | +2.7% TPS | ΔPPL = 0.00% |
| DistilGPT-2 (T4 GPU) | SeqGate | +3.4% TPS | ΔPPL = +0.51% |

## Dataset Information

### CIFAR-10
- **Source**: Canadian Institute for Advanced Research
- **URL**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Description**: 60,000 32×32 color images in 10 classes
- **Access**: Via `torchvision.datasets.CIFAR10`
- **Citation**: Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical Report, University of Toronto.

### WikiText-2
- **Source**: Salesforce Research
- **URL**: https://huggingface.co/datasets/wikitext
- **Identifier**: `wikitext-2-raw-v1`
- **Description**: Over 2 million tokens from Wikipedia Good/Featured articles
- **Access**: Via Hugging Face `datasets` library
- **Citation**: Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843.

## Code Information

### Repository Structure

```
ExNAS/
├── exnas_cnn.py                      # CNN experiments (CIFAR-10, CPU)
├── exnas_qwen_v2.py                  # Qwen2-1.5B evaluation (T4 GPU)
├── exnas_seqgate_v2_distilgpt2.py    # DistilGPT-2 evaluation (T4 GPU)
├── requirements.txt                   # Exact dependency versions
├── README.md                          # This file
└── LICENSE                            # MIT License
```

### Core Components

1. **Experiential Memory**: Stores layer-wise fingerprints (d=64), lightweight context, and timestamps
2. **Transversal Selector**: Computes neuron scores combining current evidence and memory evidence
3. **Budget Controller**: Enforces per-layer and global constraints with alignment
4. **SeqGate**: Per-sequence experiential gating with sentinel-ratio calibration

## Usage Instructions

### Installation

```bash
# Clone the repository
git clone https://github.com/ARTUS10/ExNAS.git
cd ExNAS

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

```bash
# CNN experiments on CPU
python exnas_cnn.py

# Qwen2-1.5B experiments (requires GPU)
python exnas_qwen_v2.py

# DistilGPT-2 experiments (requires GPU)
python exnas_seqgate_v2_distilgpt2.py
```

### Loading Datasets

```python
# CIFAR-10
from torchvision import datasets, transforms
cifar10 = datasets.CIFAR10(root='./data', train=False, download=True,
                           transform=transforms.ToTensor())

# WikiText-2
from datasets import load_dataset
wikitext2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
```

## Requirements

### Software Dependencies

```
Python >= 3.12
PyTorch >= 2.3.1
torchvision >= 0.18.1
transformers >= 4.41.1
datasets >= 2.20.0
numpy >= 2.1.3
scipy >= 1.13.1
tqdm >= 4.66.4
```

### Hardware Requirements

- **CNN experiments**: Any modern CPU (tested on x86-64)
- **Transformer experiments**: NVIDIA GPU with ≥16GB VRAM (tested on T4)
- **Memory**: ≥16GB RAM recommended

## Methodology

### Data Preprocessing

**CIFAR-10 (CNNs)**:
- Conversion to PyTorch tensors
- Normalization to [0,1] range
- No augmentation during evaluation

**WikiText-2 (Transformers)**:
- Streaming mode with sliding window (stride 256-512)
- Attention masks always provided
- pad_token replaced with eos_token
- Labels set to -100 for padding positions

### Evaluation Protocol

1. **Ablation study**: BASE → STATIC → STATIC+EXPERIENCE
2. **Budget sensitivity**: Grid search over bl ∈ {0.10, 0.12, 0.18} and Bg ∈ {0.05, 0.06, 0.08}
3. **Cross-architecture testing**: CNNs and Transformers

### Assessment Metrics

- **Throughput**: samples/s (CNN) or tokens/s (Transformers)
- **Wall-clock time**: Monotonic clock measurement
- **Accuracy**: Top-1 (CNNs)
- **Perplexity**: Token-level PPL with ΔPPL ≤ +1% guardrail (Transformers)
- **Active fraction**: Proportion of enabled units

### Statistical Rigor

- R=5 repeats with fixed seeds
- Mean ± 95% CI reported
- Warm-up iterations excluded from timing

## Reproducibility

### Setting Random Seeds

```python
import os
import numpy as np
import torch

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
torch.manual_seed(123)
```

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| d | 64 | Fingerprint dimension |
| K | 16 | Top-K retrieval |
| bl | 0.12 | Per-layer budget |
| Bg | 0.06 | Global budget cap |
| wcur | 0.8 | Current evidence weight |
| wmem | 0.2 | Memory evidence weight |
| align_channels | 64/128 | Kernel alignment |

### Environment Notes

On some T4 GPUs, set:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For debugging:
```bash
export CUDA_LAUNCH_BLOCKING=1
```

## Citations

If you use this code, please cite:

```
@article{lancho2025exnas,
  title={ExNAS: Experiential Neural Architecture Selection for Real-Time Inference Optimization},
  author={Lancho Rodríguez, José María},
  journal={PeerJ Computer Science (under review)},
  year={2025},
  note={Preprint available at Research Square: https://doi.org/10.21203/rs.3.rs-7378044/v1}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Contact

- **Author**: José María Lancho Rodríguez
- **Affiliation**: Independent Researcher, Fundación para la Transparencia del Software, Madrid, Spain
- **Email**: jml@josemarialancho.com
- **ORCID**: [0009-0007-9590-3163](https://orcid.org/0009-0007-9590-3163)

## Acknowledgements

AI-assisted tools (Claude by Anthropic and ChatGPT by OpenAI) were used for language editing and programming assistance. AI was not used to generate experimental results or scientific conclusions. All scientific content is the author's original work.
