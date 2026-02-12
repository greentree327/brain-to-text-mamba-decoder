"""
Brain-to-Text Mamba Decoder - Production-ready neural decoding library.

This package provides models and utilities for decoding neural signals to text,
including Mamba and GRU architectures with ensemble and language model support.
"""

from .models import (
    MambaDecoder,
    GRUDecoderBaseline,
    SoftWindowBiMamba,
)

from .utils import (
    compute_wer,
    compute_cer,
    gauss_smooth,
    remove_punctuation,
    phoneme_ids_to_text,
)

from .data_loader import (
    BrainToTextDataset,
    create_data_loader,
    collate_batch,
)

from .decoding import (
    run_single_decoding_step,
    ensemble_logit_averaging,
    ensemble_majority_vote,
    apply_test_time_augmentation,
    aggregate_tta_outputs,
    LISAEnsemble,
    beam_search_decode,
)

__version__ = "1.0.0"
__author__ = "Brain-to-Text Team"

__all__ = [
    # Models
    "MambaDecoder",
    "GRUDecoderBaseline",
    "SoftWindowBiMamba",
    # Utils
    "compute_wer",
    "compute_cer",
    "gauss_smooth",
    "remove_punctuation",
    "phoneme_ids_to_text",
    # Data
    "BrainToTextDataset",
    "create_data_loader",
    "collate_batch",
    # Decoding
    "run_single_decoding_step",
    "ensemble_logit_averaging",
    "ensemble_majority_vote",
    "apply_test_time_augmentation",
    "aggregate_tta_outputs",
    "LISAEnsemble",
    "beam_search_decode",
]
