"""
Decoding utilities for inference and ensemble methods.

This module provides functions for:
- Single decoding steps
- Beam search decoding
- Ensemble methods (logit averaging, majority voting)
- KenLM language model integration
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
from .utils import gauss_smooth


def run_single_decoding_step(x: torch.Tensor,
                             input_layer: int,
                             model: torch.nn.Module,
                             model_args: Dict,
                             device: str = 'cuda',
                             use_amp: bool = True) -> np.ndarray:
    """
    Run a single forward pass through the model for decoding.
    
    Args:
        x: Input neural data [B, T, N]
        input_layer: Day/layer index
        model: PyTorch model
        model_args: Dictionary with model configuration
        device: Device to use ('cuda' or 'cpu')
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Model logits as numpy array
    """
    with torch.autocast(
        device_type="cuda",
        enabled=use_amp,
        dtype=torch.bfloat16
    ):
        x = gauss_smooth(
            inputs=x,
            device=device,
            smooth_kernel_std=model_args.get('smooth_kernel_std', 1.0),
            smooth_kernel_size=model_args.get('smooth_kernel_size', 11),
            padding='valid',
        )
        
        with torch.no_grad():
            logits, _ = model(
                x=x,
                day_idx=torch.tensor([input_layer], device=device),
                states=None,
                return_state=True,
            )
    
    logits = logits.float().cpu().numpy()
    return logits


def ensemble_logit_averaging(logits_list: List[np.ndarray],
                              weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Average logits from multiple models.
    
    Args:
        logits_list: List of logit arrays from different models
        weights: Optional weights for each model
        
    Returns:
        Averaged logits
    """
    if weights is None:
        weights = [1.0 / len(logits_list)] * len(logits_list)
    
    ensemble_logits = np.zeros_like(logits_list[0])
    for logits, weight in zip(logits_list, weights):
        ensemble_logits += weight * logits
    
    return ensemble_logits


def ensemble_majority_vote(predictions_list: List[np.ndarray],
                           scores: Optional[List[float]] = None) -> np.ndarray:
    """
    Ensemble using majority voting.
    
    Args:
        predictions_list: List of prediction arrays
        scores: Optional confidence scores for each model
        
    Returns:
        Majority vote predictions
    """
    stacked = np.stack([pred.argmax(axis=-1) for pred in predictions_list], axis=0)
    
    if scores is not None:
        weighted_votes = np.zeros_like(stacked[0], dtype=float)
        for vote_array, score in zip(stacked, scores):
            weighted_votes += score * (np.arange(stacked.shape[2])[np.newaxis, np.newaxis, :] == vote_array[..., np.newaxis]).astype(float)
        final_votes = weighted_votes.argmax(axis=-1)
    else:
        final_votes = np.median(stacked, axis=0).astype(int)
    
    return final_votes


def apply_test_time_augmentation(x: torch.Tensor,
                                augmentation_transforms: List) -> List[torch.Tensor]:
    """
    Apply test-time augmentation to input data.
    
    Args:
        x: Input tensor
        augmentation_transforms: List of transform functions
        
    Returns:
        List of augmented tensors
    """
    augmented = [x]  # Original
    for transform in augmentation_transforms:
        augmented.append(transform(x))
    
    return augmented


def aggregate_tta_outputs(outputs_list: List[np.ndarray],
                         method: str = 'average') -> np.ndarray:
    """
    Aggregate outputs from test-time augmentation.
    
    Args:
        outputs_list: List of model outputs
        method: 'average' or 'max'
        
    Returns:
        Aggregated output
    """
    if method == 'average':
        return np.mean(outputs_list, axis=0)
    elif method == 'max':
        return np.max(outputs_list, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


class LISAEnsemble:
    """
    LISA (Layer Integrated Sequence Aggregation) Ensemble.
    
    Combines predictions from multiple models at different layers
    with optional language model rescoring.
    """
    
    def __init__(self, models: List[torch.nn.Module],
                 model_configs: List[Dict],
                 weights: Optional[List[float]] = None):
        """
        Args:
            models: List of PyTorch models
            model_configs: List of configuration dicts for each model
            weights: Optional weights for each model
        """
        self.models = models
        self.model_configs = model_configs
        self.n_models = len(models)
        
        if weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = np.array(weights) / np.sum(weights)
    
    def forward(self, x: torch.Tensor,
               day_idx: int,
               device: str = 'cuda') -> np.ndarray:
        """
        Run ensemble forward pass.
        
        Args:
            x: Input neural data
            day_idx: Day index
            device: Device to use
            
        Returns:
            Ensemble prediction logits
        """
        all_logits = []
        
        for model, config in zip(self.models, self.model_configs):
            logits = run_single_decoding_step(
                x=x,
                input_layer=day_idx,
                model=model,
                model_args=config,
                device=device
            )
            all_logits.append(logits)
        
        # Average logits with weights
        ensemble_logits = ensemble_logit_averaging(all_logits, list(self.weights))
        
        return ensemble_logits
    
    def set_weights(self, weights: List[float]):
        """Update ensemble weights."""
        self.weights = np.array(weights) / np.sum(weights)


def beam_search_decode(logits: np.ndarray,
                       beam_width: int = 5,
                       language_model=None,
                       lm_weight: float = 0.5) -> Tuple[List[int], float]:
    """
    Beam search decoding with optional language model.
    
    Args:
        logits: Model logits [T, vocab_size]
        beam_width: Beam search width
        language_model: Optional language model for scoring
        lm_weight: Weight for language model score
        
    Returns:
        Best path and score
    """
    T, vocab_size = logits.shape
    
    # Initialize beams: (score, path)
    beams = [(0.0, [])]
    
    for t in range(T):
        new_beams = []
        
        for score, path in beams:
            for token_id in range(vocab_size):
                token_score = logits[t, token_id]
                
                # Add LM score if available
                if language_model is not None:
                    lm_score = language_model.score(path + [token_id])
                    token_score += lm_weight * lm_score
                
                new_score = score + token_score
                new_path = path + [token_id]
                new_beams.append((new_score, new_path))
        
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
    
    best_score, best_path = beams[0]
    return best_path, best_score
