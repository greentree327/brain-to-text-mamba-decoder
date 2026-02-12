# Running the Brain-to-Text Mamba Decoder on Google Colab

## 🎯 Why Google Colab?

This solution uses **mamba-ssm** which can be challenging to install locally (especially on Windows). Google Colab provides:
- ✅ Pre-configured GPU environment (T4/A100)
- ✅ All dependencies install cleanly
- ✅ No local setup required
- ✅ Free GPU access

---

## 📋 Prerequisites

Before starting, you'll need:

1. **Kaggle API Credentials** (for downloading datasets/models)
   - Go to https://www.kaggle.com/settings
   - Click "Create New Token" under API section
   - Save the username and key

2. **HuggingFace Token** (for Mistral-7B models)
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "read" permissions
   - Save the token

---

## 🚀 Step-by-Step Guide

### Step 1: Push This Repository to Your GitHub

First, push this repository to your GitHub account (if you haven't already):

```bash
# In your local repository folder
git add .
git commit -m "Initial commit: Modularized Mamba+GRU decoder"
git remote add origin https://github.com/YOUR_USERNAME/brain-to-text-mamba-decoder.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 2: Open the Colab Notebook

1. Go to Google Colab: https://colab.research.google.com
2. Click **File** → **Upload notebook**
3. Navigate to the **GitHub** tab
4. Enter your repository URL: `https://github.com/YOUR_USERNAME/brain-to-text-mamba-decoder`
5. Select `notebooks/inference_colab.ipynb`

**OR** open directly via URL:
```
https://colab.research.google.com/github/YOUR_USERNAME/brain-to-text-mamba-decoder/blob/main/notebooks/inference_colab.ipynb
```

### Step 3: Configure GPU Runtime

1. Click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: GPU
3. Choose **GPU type**: T4 (free) or A100 (paid, recommended for faster inference)
4. Click **Save**

### Step 4: Update GitHub Username in Cell 3

In the first code cell (Cell 3), update the clone URL:

```python
# BEFORE:
!git clone https://github.com/YOUR_USERNAME/brain-to-text-mamba-decoder.git

# AFTER (replace with your actual username):
!git clone https://github.com/greentree327/brain-to-text-mamba-decoder.git
```

### Step 5: Run All Cells

Click **Runtime** → **Run all** and follow the prompts:

1. **Cell 3**: Repository clones from GitHub (~5 seconds)
2. **Cell 4**: Dependencies install (~5 minutes)
3. **Cell 5**: Enter your Kaggle username and API key
4. **Cell 6**: Enter your HuggingFace token
5. **Cell 7**: Datasets download (~10-15 minutes, ~20GB total)
6. **Cells 8-13**: Models load into memory (~2 minutes)
7. **Cell 16**: Full inference runs (~30-45 minutes for test set)
8. **Cell 17**: Submission CSV downloads automatically

### Step 6: Download Results

The final cell automatically downloads `submission_colab.csv` to your computer. This file contains:
- `sentence_id`: Test sample identifier
- `predicted_text`: Decoded text from neural signals

You can submit this CSV to the [Kaggle competition](https://www.kaggle.com/competitions/brain-to-text-25).

---

## 📦 What Gets Downloaded

The notebook downloads these datasets from Kaggle (total ~20GB):

| Dataset | Size | Description |
|---------|------|-------------|
| Competition test data | ~2GB | Neural recordings to decode |
| 10 Mamba model checkpoints | ~12GB | Pre-trained decoder weights |
| 4 GRU model checkpoints | ~3GB | Baseline ensemble models |
| KenLM 4-gram model | ~2GB | Language model for beam search |
| Lexicon & tokens | ~100MB | Phoneme-to-word mappings |

---

## 🏗️ Code Structure (Modularized)

The Colab notebook **imports from our modularized `src/` code**:

```python
from src.models import MambaDecoder, GRUDecoderBaseline
from src.data_sources import download_all_sources
from src.utils import compute_wer, compute_cer
```

**Key advantage**: No code duplication! All model classes are defined in `src/models.py` - the notebook just imports and uses them. This means:
- ✅ Single source of truth
- ✅ Easy to update/debug
- ✅ Professional code organization
- ✅ Testable components

---

## 🔧 Troubleshooting

### "Repository not found" error
- Make sure you've pushed to GitHub and the repository is public
- Double-check the username in the clone URL

### "Kaggle API credentials not valid"
- Verify your API key at https://www.kaggle.com/settings
- Make sure you entered both username AND key

### "Out of memory" error
- Upgrade to **A100 GPU** (Colab Pro)
- Or reduce batch size in inference loop

### "Model download failed"
- Check your Kaggle account has accepted the competition rules
- Ensure you have internet connectivity in Colab

---

## ⏱️ Expected Runtime

| Step | Duration |
|------|----------|
| Dependency installation | ~5 min |
| Dataset downloads | ~10-15 min |
| Model loading | ~2 min |
| Full test set inference | ~30-45 min |
| **Total** | **~50-60 min** |

---

## 🎓 Understanding the Architecture

While the notebook runs, you can read about the technical approach:
- **Medium Article**: [7th Place Solution Technical Writeup](https://medium.com/@jackson3b04/7th-place-solution-mamba-gru-kenlm-with-code-brain-to-text-25-00f1c69dcd0d)
- **Model Architecture**: See `src/models.py` for SoftWindow Bi-Mamba implementation
- **Ensemble Strategy**: Cell 16 shows LISA selection with Mistral-7B

---

## 📊 Expected Results

After inference completes, you should see:
- **~1,000 predictions** (test set size)
- **Strategy distribution**: ~60-70% "coherent", ~30-40% "random"
- **Sample outputs**: First 5 predictions displayed
- **Expected WER**: ~0.026-0.029 (2.6-2.9%)

---

## 💡 Next Steps

After running the baseline notebook:
1. Experiment with different beam search parameters (Cell 13)
2. Adjust LLM weight for rescoring (Cell 16: `COHERENT_LLM_WEIGHT`)
3. Try different n-gram thresholds (Cell 16: `NGRAM_THRESHOLD`)
4. Add Test-Time Augmentation (TTA) for improved accuracy

---

## ❓ Need Help?

- Check the [README.md](README.md) for architecture overview
- Review the [original Kaggle notebook](https://www.kaggle.com/code/heyyousum/slightly-tidy-version-mamba-gru-ensemble-with-tta) (monolithic version)
- Open an issue on GitHub
