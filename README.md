# ExNAS: Experiential Neural Architecture Selection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

## Overview

**ExNAS** (Experiential Neural Architecture Selection) performs real-time structural adaptation during inference using a lightweight experiential memory that updates *online*. Unlike static pruning or input-adaptive methods with frozen gating networks, ExNAS learns from routing mistakes within the same inference session.

**Paper**: *ExNAS: Experiential Neural Architecture Selection for Real-Time Inference Optimization* (PeerJ Computer Science, under review)

## Key Results

| Experiment | Key Finding |
|------------|-------------|
| **Trap Zone (CIFAR-100)** | +56pp accuracy vs DynaBERT on confident errors (94% vs 38%) |
| **CNN Online Learning** | 1,765 memory updates during inference |
| **ResNet-56 A100** | 65% slim usage, 527 online updates, 0.003ms overhead |
| **Video Processing** | 1.60-1.83x speedup via scene change detection |
| **Fingerprint Ablation** | Sentinel ratio 26,000x faster than entropy |
| **Qwen2-0.5B** | 100% safety accuracy, 0.8 layers saved per query |

## Installation

```bash
git clone https://github.com/ARTUS10/ExNAS.git
cd ExNAS
pip install -r requirements.txt
```

## Experiments

### 1. CNN Online Learning (CIFAR-10)
Demonstrates online memory updates during inference (Paper: Table 3).

```bash
python "experiments/CNN Online Learning.py"
```

**Results**: 89.88% baseline, 82.35% ExNAS inference, 25.5% computational savings, **1,765 online updates**

### 2. Trap Zone Experiment (CIFAR-100)
The key experiment showing ExNAS's advantage on "confident errors" (Paper: Tables 6, 7).

```bash
python "experiments/ExNAS vs DynaBERT Trap Zone Experiment on CIFAR-100.py"
```

**Results**: DynaBERT 38% vs ExNAS 94% on trap inputs (+56pp). DynaBERT repeats the same routing error forever; ExNAS learns from the first mistake.

### 3. ResNet-56 A100 Comparison
Direct comparison with DynaBERT, DynamicViT-style router, and static L1 pruning (Paper: Table 4).

```bash
python "experiments/ResNet-56 A100 Comparison.py"
```

**Results**:
| Method | Accuracy | Overhead | Slim % | Online |
|--------|----------|----------|--------|--------|
| Full model | 93.53% | — | 0% | No |
| Static L1 | 92.36% | 0 ms | 25% | No |
| DynaBERT | 93.58% | 0.002 ms | 50% | No |
| **ExNAS** | **92.74%** | **0.003 ms** | **65%** | **Yes (527)** |

### 4. Fingerprint Ablation
Comparison of fingerprint metrics across visual scenarios (Paper: Table 8, Figure 2).

```bash
python "experiments/Fingerprint Ablation.py"
```

**Results**: Sentinel Energy Ratio achieves 85% accuracy vs L1-Norm 78% vs Random 60%, with 26,000x speed advantage over entropy (0.001ms vs 79ms).

### 5. Video Processing
Scene change detection with adaptive model selection (Paper: Section 4.5).

```bash
python "experiments/Video Processing (Scene Change Detection).py"
```

**Results**: 1.60-1.83x speedup over DynaBERT-style static routing. Memory distance jumps from 0.01 to 2.07 at scene transitions.

### 6. Qwen2-0.5B Sentinel & Online Memory
LLM profile selection with online learning (Paper: Section 4.5).

```bash
python "experiments/Qwen2 Sentinel & Online Memory.py"
```

**Results**: 100% safety accuracy, 0.8 layers saved per query. Query "10+10=" learns correct profile after 2 iterations.

## Repository Structure

```
ExNAS/
├── experiments/
│   ├── CNN Online Learning.py                                # Table 3
│   ├── ExNAS vs DynaBERT Trap Zone Experiment on CIFAR-100.py # Tables 6, 7
│   ├── ResNet-56 A100 Comparison.py                          # Table 4
│   ├── Fingerprint Ablation.py                               # Table 8, Fig. 2
│   ├── Video Processing (Scene Change Detection).py          # Section 4.5
│   └── Qwen2 Sentinel & Online Memory.py                    # Section 4.5
├── requirements.txt
├── LICENSE
└── README.md
```

## Core Concept

The key insight is that **online memory updates** enable learning from routing mistakes within the same inference session:

1. **Static pruning**: Commits to fixed configuration at calibration time
2. **Input-adaptive (DynaBERT, DynamicViT)**: Frozen gating network after training
3. **ExNAS**: Non-parametric memory that updates online

When a routing error occurs, ExNAS stores the fingerprint with `difficulty=1.0`. If a similar input appears later, memory retrieval identifies it as risky and routes to the full model. This is impossible with frozen decision boundaries.

## Datasets

- **CIFAR-10/100**: Via `torchvision.datasets`
- **WikiText-2**: Via `datasets` library

## Hardware Requirements

- **CNN experiments**: Any modern CPU or GPU
- **Transformer experiments**: NVIDIA GPU with 8GB+ VRAM
- **A100 experiments**: NVIDIA A100 GPU

## Citation

```bibtex
@article{lancho2025exnas,
  title={ExNAS: Experiential Neural Architecture Selection for Real-Time Inference Optimization},
  author={Lancho Rodr{\'\i}guez, Jos{\'e} Mar{\'\i}a},
  journal={PeerJ Computer Science},
  year={2025},
  note={Under review}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Author

**José María Lancho Rodríguez**
Independent Researcher
Fundación para la Transparencia del Software, Madrid, Spain
Email: jml@josemarialancho.com
ORCID: [0009-0007-9590-3163](https://orcid.org/0009-0007-9590-3163)

## Acknowledgements

AI-assisted tools (Claude by Anthropic, Gemini by Google) were used for code review and documentation. All experimental results and scientific conclusions are the author's original work.
