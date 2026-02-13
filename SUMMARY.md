# Technical Implementation Summary

## Architecture Overview

This repository implements a production-ready brain-to-text decoder that achieved **7th place (top 1.5%)** in the Kaggle Brain-to-Text 2025 competition. The solution combines state-space models (Mamba) with recurrent architectures (GRU) to decode intracortical neural signals into natural language.

---

## System Design

### 1. Hybrid Neural Architecture

**SoftWindow Bi-Mamba Decoder** (`src/models.py:MambaDecoder`)
- Bidirectional Mamba2 layers with forced short-term memory bias
- Day-specific linear transformations for electrode drift compensation
- Stochastic depth regularization (drop path rate: 0.2)
- Patch-based temporal encoding (513D input → 2048D hidden)
- **Key Parameters**: d_state=64, d_conv=4, expand=2, n_layers=5

**GRU Baseline Ensemble** (`src/models.py:GRUDecoderBaseline`)
- Orthogonal weight initialization for stability
- Day-indexed parameter matrices
- Provides ensemble diversity and robustness

### 2. Ensemble Strategy

**Model Configuration**:
- **10 Mamba models** organized in 4 groups:
  - Group 1 (models 0-2): WER 0.02818
  - Group 2 (models 3-5): WER 0.02727
  - Group 3 (model 6): WER 0.02787
  - Group 4 (models 7-9): WER 0.02606 ⭐
- **4 GRU models**: Baseline ensemble for stability

**Logit Averaging**: Within-group predictions combined via arithmetic mean before beam search

### 3. Language Model Integration

**KenLM 4-gram** (Wiki + Switchboard + News corpus)
- Optimized Flashlight decoder with trie compression
- Reduced memory footprint vs. standard 5-gram models

**CTC Beam Search**:
```python
beam_size=1500, nbest=50, lm_weight=4.0, word_score=-0.5
```

### 4. Dynamic Inference Pipeline (LISA)

**Adaptive Gating Mechanism**:
```python
if ngram_score >= THRESHOLD:  # -3.76
    strategy = 'coherent'  # Use LLM rescoring
else:
    strategy = 'random'    # Use greedy decoding
```

**Mistral-7B Integration**:
- Coherence scoring via negative log-likelihood (NLL)
- Final score: `beam_score - (7.5 × llm_nll)`
- **Mistral-7B-Instruct**: LISA prompting for multi-candidate selection

### 5. Day-Specific Calibration

Neural drift compensation using normalized post-implant day feature:
```python
implant_day_normalized = (day - min_day) / (max_day - min_day)
neural_input_513 = concat([neural_512, implant_day_normalized])
```

Learnable per-day transformations:
```python
x_calibrated = activation(x @ W_day[i] + b_day[i])
```

---

## Data Pipeline

**Automated Dataset Management** (`src/data_sources.py`)

The `download_all_sources()` function handles Kaggle API integration:

| Dataset | Size | Source |
|---------|------|--------|
| Competition test data | ~2GB | `brain-to-text-25` |
| Mamba model checkpoints | ~12GB | 10 pre-trained models |
| GRU model checkpoints | ~3GB | 4 baseline models |
| KenLM 4-gram model | ~2GB | Custom-trained corpus |
| Phoneme lexicon | ~100MB | CTC token mappings |

---

## Codebase Architecture

### Modular Design

```
src/
├── models.py          # MambaDecoder, GRUDecoderBaseline classes
├── data_loader.py     # PyTorch Dataset for HDF5 neural data
├── data_sources.py    # Kaggle dataset download automation
├── decoding.py        # Beam search, ensemble methods
└── utils.py           # WER/CER metrics, preprocessing

tests/
├── test_models.py     # Model forward pass validation
└── test_utils.py      # Utility function tests

brain_to_text_colab.ipynb  # End-to-end inference pipeline
```

### Design Principles

1. **Separation of Concerns**: Model architecture decoupled from data loading
2. **Testability**: 40+ unit tests for core components
3. **Reproducibility**: All hyperparameters stored in `args.yaml` checkpoints
4. **Portability**: Runs on Colab, local GPU, or cloud instances

---

## Performance Benchmarks

### Accuracy Results

| Metric | Score |
|--------|-------|
| Word Error Rate (WER) | **0.02994** (private LB) |
| Competition Rank | **7th / 463 teams** |
| Percentile | **Top 1.5%** |

---

## Reproducibility

### Running on Google Colab

1. Open: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greentree327/brain-to-text-mamba-decoder/blob/main/brain_to_text_colab.ipynb)
2. Runtime → Change runtime type → **GPU (A100 recommended)**
3. Set up Colab Secrets (one-time):
   - Click 🔑 icon (Secrets) in left sidebar
   - Add: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`
   - Enable "Notebook access" for each
4. Run all cells
5. Outputs: `submission_colab.csv` with predictions

### Required Credentials

- **Kaggle API**: https://www.kaggle.com/settings (for dataset downloads)
- **HuggingFace Token**: https://huggingface.co/settings/tokens (for Mistral-7B)

---

## Key Innovations

1. **Hybrid Architecture**: Mamba+GRU ensemble for BCI decoding
2. **Memory Efficiency**: Optimized language model vs. baseline 5-gram
3. **Adaptive Inference**: Dynamic LLM gating for efficiency
4. **Modular Design**: Production-ready code structure

---

## References

- **Competition**: [Kaggle Brain-to-Text 2025](https://www.kaggle.com/competitions/brain-to-text-25)
- **Technical Writeup**: [Medium Article](https://medium.com/@jackson3b04/7th-place-solution-mamba-gru-kenlm-with-code-brain-to-text-25-00f1c69dcd0d)
- **Mamba Paper**: [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)
- **Original Dataset**: [Willett et al., Nature 2023](https://www.nature.com/articles/s41586-023-06377-x)
