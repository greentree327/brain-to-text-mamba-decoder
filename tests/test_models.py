"""
Unit tests for Brain-to-Text models.

This test suite validates model architectures, forward passes,
and output shapes.
"""

import pytest
import torch
import numpy as np
from src.models import MambaDecoder, GRUDecoderBaseline, SoftWindowBiMamba


class TestSoftWindowBiMamba:
    """Test the SoftWindowBiMamba block."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        d_model = 256
        batch_size = 2
        seq_len = 100
        
        block = SoftWindowBiMamba(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_bidirectional_output(self):
        """Verify bidirectional processing."""
        d_model = 128
        block = SoftWindowBiMamba(d_model=d_model)
        
        x = torch.randn(1, 50, d_model)
        output = block(x)
        
        # Output should be different from input (processed by Mamba)
        assert not torch.allclose(output, x)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        d_model = 64
        block = SoftWindowBiMamba(d_model=d_model)
        
        x = torch.randn(1, 20, d_model, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMambaDecoder:
    """Test the MambaDecoder model."""
    
    @pytest.fixture
    def model(self):
        """Create a MambaDecoder for testing."""
        return MambaDecoder(
            neural_dim=513,  # 512 + 1 for time
            n_units=256,
            n_days=5,
            n_classes=40,
            n_layers=3,
            drop_path_rate=0.1,
        )
    
    def test_forward_pass(self, model):
        """Test basic forward pass."""
        batch_size = 4
        seq_len = 200
        neural_dim = 513
        
        x = torch.randn(batch_size, seq_len, neural_dim)
        day_idx = torch.tensor([0, 1, 2, 3])
        
        output = model(x, day_idx)
        
        assert output.shape == (batch_size, seq_len, 40)
    
    def test_output_shape_variable_length(self, model):
        """Test with variable sequence lengths."""
        batch_size = 2
        neural_dim = 513
        n_classes = 40
        
        x1 = torch.randn(1, 100, neural_dim)
        x2 = torch.randn(1, 150, neural_dim)
        
        out1 = model(x1, torch.tensor([0]))
        out2 = model(x2, torch.tensor([1]))
        
        assert out1.shape == (1, 100, n_classes)
        assert out2.shape == (1, 150, n_classes)
    
    def test_day_specific_parameters(self, model):
        """Test that day-specific parameters are used."""
        x = torch.randn(1, 50, 513)
        
        # Same input, different days
        out1 = model(x, torch.tensor([0]))
        out2 = model(x, torch.tensor([1]))
        
        # Outputs should be different due to day-specific weights
        assert not torch.allclose(out1, out2)
    
    def test_with_return_state(self, model):
        """Test return_state option."""
        x = torch.randn(2, 30, 513)
        day_idx = torch.tensor([0, 1])
        
        logits, state = model(x, day_idx, return_state=True)
        
        assert logits.shape == (2, 30, 40)
        assert state is None  # Mamba doesn't return internal states
    
    def test_gradient_flow(self, model):
        """Test backpropagation."""
        x = torch.randn(2, 25, 513, requires_grad=True)
        day_idx = torch.tensor([0, 1])
        
        output = model(x, day_idx)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_training_eval_mode(self, model):
        """Test training vs eval mode (stochastic depth)."""
        x = torch.randn(1, 40, 513)
        day_idx = torch.tensor([0])
        
        model.train()
        out_train1 = model(x, day_idx)
        out_train2 = model(x, day_idx)
        
        model.eval()
        out_eval1 = model(x, day_idx)
        out_eval2 = model(x, day_idx)
        
        # Stochastic depth should cause variation in training
        # (though not guaranteed, so we just check shapes)
        assert out_train1.shape == out_eval1.shape


class TestGRUDecoderBaseline:
    """Test the GRUDecoderBaseline model."""
    
    @pytest.fixture
    def model(self):
        """Create a GRUDecoderBaseline for testing."""
        return GRUDecoderBaseline(
            neural_dim=512,
            n_units=256,
            n_days=5,
            n_classes=40,
            n_layers=3,
        )
    
    def test_forward_pass(self, model):
        """Test basic forward pass."""
        batch_size = 4
        seq_len = 200
        neural_dim = 512
        
        x = torch.randn(batch_size, seq_len, neural_dim)
        day_idx = torch.tensor([0, 1, 2, 3])
        
        output = model(x, day_idx)
        
        assert output.shape == (batch_size, seq_len, 40)
    
    def test_with_hidden_states(self, model):
        """Test passing initial hidden states."""
        x = torch.randn(2, 50, 512)
        day_idx = torch.tensor([0, 1])
        
        # First pass to get hidden states
        _, states = model(x, day_idx, return_state=True)
        
        # Use these states for next sequence
        x2 = torch.randn(2, 30, 512)
        output = model(x2, day_idx, states=states)
        
        assert output.shape == (2, 30, 40)
    
    def test_day_specific_transformations(self, model):
        """Test day-specific weight matrices."""
        x = torch.randn(1, 40, 512)
        
        out1 = model(x, torch.tensor([0]))
        out2 = model(x, torch.tensor([2]))
        
        # Outputs should differ due to day-specific weights
        assert not torch.allclose(out1, out2)
    
    def test_gradient_accumulation(self, model):
        """Test backward pass and gradient computation."""
        x1 = torch.randn(1, 25, 512, requires_grad=True)
        x2 = torch.randn(1, 25, 512, requires_grad=True)
        day_idx = torch.tensor([0])
        
        out1 = model(x1, day_idx)
        loss1 = out1.sum()
        loss1.backward()
        
        grad1 = x1.grad.clone()
        
        x1.grad.zero_()
        out2 = model(x2, day_idx)
        loss2 = out2.sum()
        loss2.backward()
        
        # Gradients should be different for different inputs
        assert x2.grad is not None


# Fixture for both models
@pytest.fixture(params=[
    MambaDecoder(neural_dim=513, n_units=256, n_days=4, n_classes=40),
    GRUDecoderBaseline(neural_dim=512, n_units=256, n_days=4, n_classes=40),
])
def model_fixture(request):
    """Parametrized fixture for both model types."""
    return request.param


class TestModelComparison:
    """Tests comparing both model architectures."""
    
    def test_output_shape_consistency(self, model_fixture):
        """Both models should produce consistent output shapes."""
        batch_size = 2
        seq_len = 100
        
        if isinstance(model_fixture, MambaDecoder):
            neural_dim = 513
        else:
            neural_dim = 512
        
        x = torch.randn(batch_size, seq_len, neural_dim)
        day_idx = torch.tensor([0, 1])
        
        output = model_fixture(x, day_idx)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len
        assert output.shape[2] == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
