"""
Unit tests for utility functions.
"""

import pytest
import torch
import numpy as np
from src.utils import compute_wer, compute_cer, gauss_smooth, remove_punctuation


class TestMetrics:
    """Test metric computation functions."""
    
    def test_compute_wer_identical(self):
        """Test WER with identical strings."""
        ref = "hello world"
        hyp = "hello world"
        wer = compute_wer(ref, hyp)
        assert wer == 0.0
    
    def test_compute_wer_substitution(self):
        """Test WER with word substitutions."""
        ref = "the dog runs"
        hyp = "the cat runs"
        wer = compute_wer(ref, hyp)
        assert wer == 1.0 / 3.0  # 1 substitution out of 3 words
    
    def test_compute_wer_insertion_deletion(self):
        """Test WER with insertions and deletions."""
        ref = "hello world"
        hyp = "hello beautiful world"
        wer = compute_wer(ref, hyp)
        # 1 insertion out of 2 reference words
        expected = 1 / 2
        assert abs(wer - expected) < 0.01
    
    def test_compute_cer_identical(self):
        """Test CER with identical strings."""
        ref = "hello"
        hyp = "hello"
        cer = compute_cer(ref, hyp)
        assert cer == 0.0
    
    def test_compute_cer_substitution(self):
        """Test CER with character substitutions."""
        ref = "hello"
        hyp = "halla"
        cer = compute_cer(ref, hyp)
        # 2 substitutions out of 5 characters
        assert abs(cer - 2/5) < 0.01
    
    def test_compute_cer_empty_string(self):
        """Test CER with empty reference."""
        ref = ""
        hyp = "hello"
        cer = compute_cer(ref, hyp)
        assert cer == 0.0


class TestTextProcessing:
    """Test text processing functions."""
    
    def test_remove_punctuation_basic(self):
        """Test basic punctuation removal."""
        text = "Hello, world!"
        result = remove_punctuation(text)
        assert result == "hello world"
    
    def test_remove_punctuation_apostrophe(self):
        """Test apostrophe handling."""
        text = "don't worry"
        result = remove_punctuation(text)
        assert "don't" in result or "dont" in result
    
    def test_remove_punctuation_hyphens(self):
        """Test hyphen handling."""
        text = "one-two three--four"
        result = remove_punctuation(text)
        assert "one" in result and "two" in result


class TestGaussianSmoothing:
    """Test Gaussian smoothing function."""
    
    def test_smooth_output_shape(self):
        """Test that smoothing preserves shape."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x = torch.randn(2, 100, 32, device=device)
        smoothed = gauss_smooth(
            x,
            smooth_kernel_std=1.0,
            smooth_kernel_size=5,
            device=device
        )
        
        # Output might be slightly shorter due to valid padding
        assert smoothed.shape[0] == x.shape[0]  # Batch
        assert smoothed.shape[2] == x.shape[2]  # Features
    
    def test_smooth_reduces_variance(self):
        """Test that smoothing reduces variance."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x = torch.randn(1, 200, 16, device=device)
        smoothed = gauss_smooth(
            x,
            smooth_kernel_std=2.0,
            smooth_kernel_size=11,
            device=device
        )
        
        # Smoothed data should have lower variance
        orig_var = x.var().item()
        smooth_var = smoothed.var().item()
        
        assert smooth_var <= orig_var  # Smoothing reduces or maintains variance
    
    def test_smooth_different_kwargs(self, device):
        """Test smoothing with different parameters."""
        x = torch.randn(1, 100, 16)
        
        # Strong smoothing
        smooth_strong = gauss_smooth(x, smooth_kernel_std=3.0, smooth_kernel_size=15)
        
        # Weak smoothing
        smooth_weak = gauss_smooth(x, smooth_kernel_std=0.5, smooth_kernel_size=5)
        
        # Both should return valid outputs
        assert smooth_strong is not None
        assert smooth_weak is not None
        assert smooth_strong.shape[2] == x.shape[2]
        assert smooth_weak.shape[2] == x.shape[2]


class TestMetricEdgeCases:
    """Test edge cases for metric functions."""
    
    def test_wer_empty_reference(self):
        """WER with empty reference."""
        wer = compute_wer("", "hello")
        assert wer == 0.0
    
    def test_wer_single_word(self):
        """WER with single word."""
        wer = compute_wer("hello", "hello")
        assert wer == 0.0
        
        wer = compute_wer("hello", "world")
        assert wer == 1.0
    
    def test_cer_long_string(self):
        """CER with long strings."""
        ref = "a" * 1000
        hyp = "a" * 1000
        cer = compute_cer(ref, hyp)
        assert cer == 0.0
    
    def test_remove_punctuation_numbers(self):
        """Test that numbers are removed by remove_punctuation."""
        text = "hello 123 world"
        result = remove_punctuation(text)
        assert "123" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
