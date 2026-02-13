# ==============================================================================
# IMPORTS.PY - Centralized Imports for Brain-to-Text Colab Notebook
# ==============================================================================
# This file contains all necessary imports and pip installations for the
# 7th place Brain-to-Text 2025 Kaggle solution.
# Import this module into the notebook with: exec(open('imports.py').read())

import os
import sys
import time

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_progress_steps = [
    "Install required packages",
    "Import core ML packages",
    "Import utilities and config",
    "Import API clients",
    "Import Mamba",
    "Check Colab helpers",
]

def _progress_iter():
    if tqdm is None:
        for i, step in enumerate(_progress_steps, start=1):
            print(f"[{i}/{len(_progress_steps)}] {step}...")
            yield
    else:
        for _ in tqdm(_progress_steps, desc="imports.py", ncols=80):
            yield

# ============================================================================
# 1. INSTALL REQUIRED PACKAGES (Colab only)
# ============================================================================

_progress = _progress_iter()
next(_progress)
os.system("pip install -q causal-conv1d>=1.2.0")
os.system("pip install -q mamba-ssm")

# ============================================================================
# 2. CORE ML & DATA PROCESSING
# ============================================================================

next(_progress)
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 3. UTILITIES & CONFIGURATION
# ============================================================================

next(_progress)
from omegaconf import OmegaConf
import editdistance
import argparse

# ============================================================================
# 4. API & AUTHENTICATION
# ============================================================================

next(_progress)
import kagglehub
from huggingface_hub import notebook_login

# ============================================================================
# 5. MAMBA & ML FRAMEWORKS
# ============================================================================

next(_progress)
from mamba_ssm import Mamba2

# ============================================================================
# 6. MISC
# ============================================================================

next(_progress)
try:
    from google.colab import drive
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("✅ All imports successful!")
