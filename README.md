# Brain-to-Text Mamba Decoder
**7th Place Solution** | Kaggle Brain-to-Text Benchmark 2025

| Metric | Performance |
| :--- | :--- |
| **Competition Rank** | **7th / 463 teams** (Top 1.5%) |
| **Word Error Rate (WER)** | **0.02994** (2.9%, private LB) |
| **Architecture** | Hybrid SoftWindow Bi-Mamba + GRU + LISA Ensemble |

<p align="left">
  <img src="https://github.com/user-attachments/assets/73f3b2c6-217f-447d-ad70-b465f307948d" width="350" alt="7th Place Achievement Card">
  <br>
  <em>Fig 1. Final Results: <strong>Rank 7th out of 463 teams</strong> (Top 1.5%) in the Brain-to-Text '25 Challenge.</em>
</p>

---

## 📖 Overview

This repository contains the original submission notebook for the [Kaggle Brain-to-Text 2025 Competition](https://www.kaggle.com/competitions/brain-to-text-25). The solution achieved **7th place** (top 1.5% of 463 teams).

### What This Repo Provides

- **Original submission notebook** used for inference
- **Colab-first workflow** with minimal setup
- **Technical Writeup**: [Medium Article](https://medium.com/@jackson3b04/7th-place-solution-mamba-gru-kenlm-with-code-brain-to-text-25-00f1c69dcd0d)

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

### 2. Novel Memory-Optimized n-gram Language Model

**KenLM 4-gram** (14GB vs. 300GB 5-gram baseline)
- Custom-trained on Wiki + Switchboard + News corpus
- Flashlight decoder with optimized trie compression
- 15× VRAM memory reduction

### 3. Dynamic Inference Pipeline (LISA)

**Adaptive Gating Mechanism:**
```python
if ngram_score >= -3.76:  # Coherent sentence detected
    strategy = 'LLM_rescoring'  # Use Mistral-7B
else:
    strategy = 'greedy_decode'  # Skip expensive LLM
```

**Result**: Reduced inference latency via adaptive gating

---

## 🚀 Quick Start (Colab Only)

Run the complete pipeline on Google Colab with zero local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greentree327/brain-to-text-mamba-decoder/blob/main/slightly_tidy_version_Mamba_GRU_LISA_Ensemble_with_TTA%20(2).ipynb)

**Setup** (one-time, before first run):

1. **Get API credentials**:
   - **Kaggle**: https://www.kaggle.com/settings → "Create New Token" → Save username + key
   - **HuggingFace**: https://huggingface.co/settings/tokens → "New token" (read access)

2. **Add to Colab Secrets** (stores credentials securely):
   - In Colab: Click 🔑 icon (left sidebar) → "Secrets"
   - Add secrets:
     - `KAGGLE_USERNAME` = your Kaggle username
     - `KAGGLE_KEY` = your Kaggle API key  
     - `HF_TOKEN` = your HuggingFace token
   - Enable "Notebook access" for each secret

3. **Run the notebook**:
   - Runtime → Change runtime type → Select **GPU**
   - Runtime → Run all
   - Credentials load automatically from secrets

---

## 🔗 Links

- **Competition**: [Kaggle Brain-to-Text 2025](https://www.kaggle.com/competitions/brain-to-text-25/leaderboard)

- **Original Dataset Paper**: [Willett et al., Nature 2023](https://www.nature.com/articles/s41586-023-06377-x)

---

## 🙏 Acknowledgments

- Kaggle and competition organizers  
