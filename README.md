# Brain-to-Text Mamba Decoder
**7th Place Solution** | Kaggle Brain-to-Text Benchmark 2025

| Metric | Performance |
| :--- | :--- |
| **Competition Rank** | **7th / 463 teams** (Top 1.5%) |
| **Word Error Rate (WER)** | **0.026** (2.6%) |
| **Model Size (RAM)** | Optimized (15× reduction from baseline) |
| **Inference Hardware** | NVIDIA A100 (80GB) |
| **Architecture** | Hybrid SoftWindow Bi-Mamba + GRU + LISA Ensemble |

<p align="left">
  <img src="https://github.com/user-attachments/assets/73f3b2c6-217f-447d-ad70-b465f307948d" width="350" alt="7th Place Achievement Card">
  <br>
  <em>Fig 1. Final Results: <strong>Rank 7th out of 463 teams</strong> (Top 1.5%) in the Brain-to-Text '25 Challenge.</em>
</p>

---

## 📖 Overview

This repository implements a production-ready brain-computer interface decoder that converts intracortical neural signals ( microelectrode array recordings) into natural language text. The solution achieved **7th place** in the [Kaggle Brain-to-Text 2025 Competition](https://www.kaggle.com/competitions/brain-to-text-25), ranking in the **top 1.5%** of 463 teams.

### Key Innovation

Unlike top-heavy solutions relying on massive 5-gram language models (>300GB RAM), this approach prioritizes **production viability** for clinical deployment:

- **Memory Optimization**: Compact footprint vs. 300GB baseline  
- **Hybrid Architecture**: Mamba (state-space models) + GRU (recurrent networks)  
- **Adaptive Inference**: Dynamic LLM gating reduces compute by ~40%  
- **Clinical-Grade**: Designed for real-time BCI hardware constraints  

---

## 🏗️ System Architecture

### 1. Hybrid Neural Decoder

**SoftWindow Bi-Mamba** (State Space Models)
- Captures long-range temporal dependencies in neural signals
- Bidirectional Mamba2 layers with forced short-term memory bias
- Day-specific linear transformations for electrode drift compensation

**GRU Baseline Ensemble**
- Provides robustness and ensemble diversity
- Orthogonal weight initialization for training stability
- Handles short-term acoustic feature extraction

### 2. Memory-Optimized Language Model

**KenLM 4-gram** (Optimized vs. 300GB)
- Custom-trained on Wiki + Switchboard + News corpus
- Flashlight decoder with optimized trie compression
- 98% perplexity retention with 15× memory reduction

### 3. Dynamic Inference Pipeline (LISA)

**Adaptive Gating Mechanism:**
```python
if ngram_score >= -3.76:  # Coherent sentence detected
    strategy = 'LLM_rescoring'  # Use Mistral-7B
else:
    strategy = 'greedy_decode'  # Skip expensive LLM
```

**Result**: ~40% reduction in average inference latency

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

Run the complete pipeline on Google Colab with zero local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greentree327/brain-to-text-mamba-decoder/blob/main/notebooks/inference_colab.ipynb)

**Features:**
- ✅ Pre-configured GPU environment (T4/A100)
- ✅ All dependencies installed automatically
- ✅ Modularized code imported from `src/`
- ✅ Credentials handled via secure prompts

See [SUMMARY.md](SUMMARY.md) for technical implementation details.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/greentree327/brain-to-text-mamba-decoder.git
cd brain-to-text-mamba-decoder

# Install Python dependencies
pip install -r requirements.txt
pip install -e .

# Install mamba-ssm (Windows requires conda)
conda install -c conda-forge mamba-ssm causal-conv1d
```

---

## 📁 Repository Structure

```
brain-to-text-mamba-decoder/
├── src/                          # Core implementation
│   ├── models.py                 # MambaDecoder, GRUDecoderBaseline
│   ├── data_loader.py            # PyTorch Dataset for HDF5 neural data
│   ├── data_sources.py           # Kaggle API dataset automation
│   ├── decoding.py               # Beam search, ensemble methods
│   └── utils.py                  # WER/CER metrics, preprocessing
├── tests/                        # Unit tests (40+ test cases)
│   ├── test_models.py            # Model forward pass validation
│   └── test_utils.py             # Utility function tests
├── notebooks/
│   └── inference_colab.ipynb     # End-to-end inference pipeline
├── requirements.txt              # Python dependencies
├── setup.py                      # Package configuration
├── README.md                     # This file
└── SUMMARY.md                    # Technical deep dive
```

---

## 🎯 Performance Benchmarks

### Competition Results

| Metric | Score |
|--------|-------|
| **Final Rank** | **7th / 463 teams** |
| **Percentile** | **Top 1.5%** |
| **Word Error Rate (WER)** | **0.02994** (2.9%) |



---

## 🔗 Links

- **Competition**: [Kaggle Brain-to-Text 2025](https://www.kaggle.com/competitions/brain-to-text-25/leaderboard)
- **Technical Writeup**: [Medium Article](https://medium.com/@jackson3b04/7th-place-solution-mamba-gru-kenlm-with-code-brain-to-text-25-00f1c69dcd0d)
- **Implementation Details**: [SUMMARY.md](SUMMARY.md)
- **Original Dataset Paper**: [Willett et al., Nature 2023](https://www.nature.com/articles/s41586-023-06377-x)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- Kaggle and competition organizers  
