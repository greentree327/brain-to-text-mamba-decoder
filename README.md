# Brain-to-Text-Mamba-Decoder
### 7th Place Solution | Kaggle Brain-to-Text Benchmark '25

| Metric | Performance |
| :--- | :--- |
| **Global Rank** | **7th / 500+** |
| **Word Error Rate (WER)** | **0.029** (2.9%) |
| **Model Size (RAM)** | **19GB** (Optimized down from ~300GB) |
| **Inference Hardware** | Single NVIDIA A100 High-Ram (80GB) |
| **Architecture** | Hybrid SoftWindow Bi-Mamba + GRU + LISA Ensemble |

<p align="left">
  <img src="https://github.com/user-attachments/assets/73f3b2c6-217f-447d-ad70-b465f307948d" width="350" alt="7th Place Achievement Card">
  <br>
  <em>Fig 1. Final Results: <strong>Rank 7th out of 463 teams</strong> (Top 1.5%) in the Brain-to-Text '25 Challenge.</em>
</p>


## 📖 Overview
This repository contains the inference pipeline for our **7th Place Solution** in the [Kaggle Brain-to-Text 2025 Competition](https://www.kaggle.com/competitions/brain-to-text-25). 

The challenge involved decoding neural signals (intracortical microelectrode arrays) into text in real-time. Unlike top-heavy solutions that relied on massive 5-gram language models (>300GB RAM), our approach focused on **production viability**. We engineered a resource-efficient **Mamba-GRU hybrid** that runs on a single GPU with **15x less memory** usage than the baseline, making it suitable for clinical-grade hardware constraints.

**🔗 Links:**
- [Competition Leaderboard](https://www.kaggle.com/competitions/brain-to-text-25/leaderboard)
- [Technical Deep Dive (Medium Article)](https://medium.com/@jackson3b04/7th-place-solution-mamba-gru-kenlm-with-code-brain-to-text-25-00f1c69dcd0d)

---

## 🏗️ System Architecture

### 1. Hybrid Neural Decoder (SoftWindow Bi-Mamba + GRU)
We leveraged the complementary strengths of two architectures:
- **Mamba (State Space Models):** Captures long-range semantic dependencies and co-articulation effects in speech.
- **GRU (Gated Recurrent Units):** Provides stability for short-term acoustic modeling and local feature extraction.

### 2. Memory-Optimized KenLM (19GB vs 300GB)
Standard 5-gram language models require ~300GB RAM, which is infeasible for portable BCI systems. We:
- Re-compiled a **custom Flashlight decoder** with optimized trie structures.
- Pruned the n-gram model to fit entirely within **19GB RAM** while maintaining 98% of the original perplexity performance.

### 3. Dynamic Inference Gating (LISA)
To save compute, we implemented a **Context-Aware Inference Pipeline (LISA)**:
- **Signal Confidence Check:** The system first evaluates the N-gram perplexity of the greedy decoding.
- **Conditional Rescoring:** Only low-confidence predictions trigger the expensive Large Language Model (Mistral-7B) rescoring step.
- **Result:** Drastic reduction in average inference latency without sacrificing accuracy on difficult sentences.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/greentree327/brain-to-text-mamba-decoder.git
cd brain-to-text-mamba-decoder

# Install dependencies
pip install -r requirements.txt
