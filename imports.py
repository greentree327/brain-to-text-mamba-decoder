# ==============================================================================
# IMPORTS.PY - Centralized Imports for Brain-to-Text Colab Notebook
# ==============================================================================
# This file contains all necessary imports and pip installations for the
# 7th place Brain-to-Text 2025 Kaggle solution.
# Import this module into the notebook with: exec(open('imports.py').read())

import os
import sys
import time

# ============================================================================
# 1. INSTALL REQUIRED PACKAGES (Colab only)
# ============================================================================

print("📦 Installing required packages...")
os.system("pip install -q causal-conv1d>=1.2.0")
os.system("pip install -q mamba-ssm")

# ============================================================================
# 2. CORE ML & DATA PROCESSING
# ============================================================================

import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 3. UTILITIES & CONFIGURATION
# ============================================================================

from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

# ============================================================================
# 4. API & AUTHENTICATION
# ============================================================================

import kagglehub
from huggingface_hub import notebook_login

# ============================================================================
# 5. MAMBA & ML FRAMEWORKS
# ============================================================================

from mamba_ssm import Mamba2

# ============================================================================
# 6. MISC
# ============================================================================

try:
    from google.colab import drive
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("✅ All imports successful!")
