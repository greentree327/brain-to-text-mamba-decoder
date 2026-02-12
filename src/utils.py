"""
Utility functions for data processing, metrics, and smoothing.

This module contains helper functions for:
- Computing Word Error Rate (WER) and Character Error Rate (CER)
- Signal smoothing (Gaussian filtering)
- Text preprocessing
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import re
import editdistance


# Phoneme to ID mapping
PHONEME_MAP = {
    'BLANK': 0,
    'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5,
    'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10,
    'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15,
    'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20,
    'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25,
    'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30,
    'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35,
    'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39,
    ' | ': 40,
}

LOGIT_TO_PHONEME = list(PHONEME_MAP.keys())


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        
    Returns:
        WER as a float (0.0 to 1.0)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    distance = editdistance.eval(ref_words, hyp_words)
    wer = distance / len(ref_words) if len(ref_words) > 0 else 0.0
    
    return wer


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) between reference and hypothesis.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        
    Returns:
        CER as a float (0.0 to 1.0)
    """
    distance = editdistance.eval(reference, hypothesis)
    cer = distance / len(reference) if len(reference) > 0 else 0.0
    
    return cer


def gauss_smooth(inputs: torch.Tensor,
                 smooth_kernel_std: float = 1.0,
                 smooth_kernel_size: int = 11,
                 padding: str = 'same',
                 device: str = 'cuda') -> torch.Tensor:
    """
    Apply 1D Gaussian smoothing to time series data.

    Args:
        inputs: [B, T, N] tensor (batch, time, features)
        smooth_kernel_std: Standard deviation of Gaussian kernel
        smooth_kernel_size: Size of the smoothing kernel
        padding: 'same' or 'valid'
        device: Device to use for computation

    Returns:
        Smoothed [B, T, N] tensor
    """
    # Create Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gauss_kernel = gaussian_filter1d(inp, smooth_kernel_std)
    
    # Threshold and normalize
    valid_idx = np.argwhere(gauss_kernel > 0.01)
    gauss_kernel = gauss_kernel[valid_idx]
    gauss_kernel = np.squeeze(gauss_kernel / np.sum(gauss_kernel))

    # Convert to tensor
    gauss_kernel = torch.tensor(gauss_kernel, dtype=torch.float32, device=device)
    gauss_kernel = gauss_kernel.view(1, 1, -1)

    # Prepare for convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gauss_kernel = gauss_kernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Apply convolution
    smoothed = torch.nn.functional.conv1d(inputs, gauss_kernel, padding=padding, groups=C)
    
    return smoothed.permute(0, 2, 1)  # [B, T, C]


def remove_punctuation(sentence: str) -> str:
    """
    Remove punctuation and normalize text.
    
    Args:
        sentence: Input text
        
    Returns:
        Cleaned text
    """
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])
    
    return sentence


def extract_transcription(input_ids: np.ndarray) -> str:
    """
    Extract transcription string from phoneme ID sequence.
    
    Args:
        input_ids: Array of phoneme IDs
        
    Returns:
        Decoded transcription string
    """
    end_idx = np.argwhere(input_ids == 0)
    if len(end_idx) > 0:
        end_idx = end_idx[0, 0]
    else:
        end_idx = len(input_ids)
    
    trans = ''
    for c in range(end_idx):
        trans += chr(input_ids[c])
    
    return trans


def phoneme_ids_to_text(phoneme_ids: np.ndarray) -> str:
    """
    Convert phoneme IDs to readable phoneme string.
    
    Args:
        phoneme_ids: Array of phoneme IDs
        
    Returns:
        Phoneme string representation
    """
    phoneme_strings = []
    for pid in phoneme_ids:
        if pid < len(LOGIT_TO_PHONEME):
            phoneme_strings.append(LOGIT_TO_PHONEME[pid])
    
    return ' '.join(phoneme_strings)
