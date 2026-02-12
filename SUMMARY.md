# How to Execute From Scratch (Ubuntu / WSL)

## Step 1: Install

```bash
conda create -n brain_to_text python=3.10
conda activate brain_to_text
pip install -r requirements.txt
pip install -e . --no-deps
conda install -c conda-forge mamba-ssm causal-conv1d
```

## Step 2: Run

```bash
jupyter notebook notebooks/inference.ipynb
```

In the notebook:
- Run Cell 2 to login to Kaggle and download data/models
- Or update paths in Cells 3 and 5, then run all cells

For non-interactive logins, edit the `.env` file in the repo root (same folder as `SUMMARY.md`) and add:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
HUGGINGFACE_TOKEN=your_hf_token
```
