"""
PyTorch models for Brain-to-Text decoding.

This module contains the Mamba, GRU, and hybrid model architectures
used for decoding neural signals to text.
"""

import torch
from torch import nn
from mamba_ssm import Mamba2
from typing import Optional, Tuple


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping a sample
        training: Whether in training mode
        scale_by_keep: Scale remaining values by keep probability
        
    Returns:
        Tensor with stochastic depth applied
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class SoftWindowBiMamba(nn.Module):
    """
    Bidirectional Mamba 2 Block with Soft Windowing.
    
    This module processes sequences in both forward and backward directions
    with Mamba 2 blocks, designed to focus on short-term memory.
    """
    
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dt_min=0.05, dt_max=1.0):
        super().__init__()
        
        self.fwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        
        self.bwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        
        self._force_short_memory_bias(self.fwd)
        self._force_short_memory_bias(self.bwd)

    def _force_short_memory_bias(self, mamba_layer):
        """Force the Mamba layer to have shorter memory by biasing dt."""
        if hasattr(mamba_layer, 'dt_bias'):
            with torch.no_grad():
                mamba_layer.dt_bias.add_(1.0)
        elif hasattr(mamba_layer, 'dt_proj'):
            with torch.no_grad():
                mamba_layer.dt_proj.bias.add_(1.0)

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Dim]
            
        Returns:
            Combined forward and backward pass output
        """
        out_fwd = self.fwd(x)
        x_rev = torch.flip(x, dims=[1])
        out_bwd = self.bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        return out_fwd + out_bwd


class MambaDecoder(nn.Module):
    """
    Mamba-based decoder for neural-to-text decoding.
    
    Combines day-specific linear layers, Mamba backbone with stochastic depth,
    and a classification head.
    """
    
    def __init__(self,
                 neural_dim: int,
                 n_units: int,
                 n_days: int,
                 n_classes: int,
                 rnn_dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 n_layers: int = 5,
                 patch_size: int = 0,
                 patch_stride: int = 0,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_min: float = 0.025,
                 drop_path_rate: float = 0.2,
                 proj_intermediate_dim: int = 4096,
                 proj_intermediate_dropout: float = 0.3,
                 final_dropout: float = 0.4):
        super(MambaDecoder, self).__init__()

        self.neural_dim_total = neural_dim
        self.n_neural_chans = neural_dim - 1
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Day-specific layers
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.n_neural_chans)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.n_neural_chans)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        # Projection layer
        self.input_size = self.neural_dim_total
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, proj_intermediate_dim),
            nn.Softsign(),
            nn.Dropout(proj_intermediate_dropout),
            nn.Linear(proj_intermediate_dim, self.n_units)
        )

        # Mamba backbone
        self.layers = nn.ModuleList([
            SoftWindowBiMamba(
                d_model=n_units,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_min=dt_min
            ) for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(self.n_units) for _ in range(self.n_layers)
        ])

        # Stochastic depth
        self.drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, n_layers)
        ]

        self.dropout = nn.Dropout(final_dropout)

        # Output head
        self.out = nn.Linear(self.n_units, self.n_classes)

        # Initialize weights
        for layer in self.input_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        Args:
            x: [Batch, Time, Features+1]
            day_idx: Day index for each sample in batch
            states: Optional initial states
            return_state: Whether to return states
            
        Returns:
            logits or (logits, states)
        """
        # Split input
        x_neural = x[:, :, :-1]
        x_time = x[:, :, -1:]

        # Apply day-specific transformation
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x_neural = torch.einsum("btd,bdk->btk", x_neural, day_weights) + day_biases
        x_neural = self.day_layer_activation(x_neural)
        x = torch.cat([x_neural, x_time], dim=-1)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # Patching
        if self.patch_size > 0:
            x = x.unsqueeze(1).permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Mamba backbone
        x = self.input_proj(x)

        for i, (norm, layer) in enumerate(zip(self.norms, self.layers)):
            x_norm = norm(x)
            layer_out = layer(x_norm)
            layer_out = drop_path(layer_out, self.drop_path_rates[i], self.training)
            x = x + layer_out

        # Output
        x = self.dropout(x)
        logits = self.out(x)

        if return_state:
            return logits, None

        return logits


class GRUDecoderBaseline(nn.Module):
    """
    GRU-based decoder for neural-to-text decoding.
    
    Combines day-specific linear layers, GRU backbone,
    and a classification head.
    """
    
    def __init__(self,
                 neural_dim: int,
                 n_units: int,
                 n_days: int,
                 n_classes: int,
                 rnn_dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 n_layers: int = 5,
                 patch_size: int = 0,
                 patch_stride: int = 0):
        super(GRUDecoderBaseline, self).__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days
        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Day-specific layers
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # GRU
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )

        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Output head
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden state
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        Args:
            x: [Batch, Time, Neural_dim]
            day_idx: Day index for each sample in batch
            states: Optional initial states
            return_state: Whether to return states
            
        Returns:
            logits or (logits, states)
        """
        # Apply day-specific transformation
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # Patching
        if self.patch_size > 0:
            x = x.unsqueeze(1)
            x = x.permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2)
            x_unfold = x_unfold.permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Initialize hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # GRU forward pass
        output, hidden_states = self.gru(x, states)

        # Output
        logits = self.out(output)

        if return_state:
            return logits, hidden_states

        return logits
