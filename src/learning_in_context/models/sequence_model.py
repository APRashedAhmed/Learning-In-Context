"""
Self-contained SequenceModel implementation for In-Context-CPD.

This module contains the core model components extracted from the timescales repository,
making the In-Context-CPD pipeline independent from external dependencies while maintaining
exact behavioral compatibility.
"""

import functools
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class PartialSoftmax(nn.Module):
    """Applies softmax activation to a specified slice of the last dimension
    of the input tensor.
    """

    def __init__(self, start: int):
        super().__init__()
        self.start = start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice the input tensor into three parts:
        # before the slice, the slice, and after the slice.
        x[..., self.start :] = torch.softmax(x[..., self.start :], dim=-1)
        return x


# ============================================================================
# RECURRENT MODULES
# ============================================================================

class AbstractRecurrentModule(nn.Module, ABC):
    """Abstract base class for recurrent modules."""
    
    @abstractmethod
    def forward(self, x, hx=None):
        """Standard forward pass."""
        pass
        
    @abstractmethod
    def forward_all_states(self, x, hx=None, interventions=None, alphas=None):
        """Forward pass returning all intermediate states."""
        pass


class LSTM_V2(AbstractRecurrentModule):
    """
    Multi-layer LSTM module with dual functionality.

    This module provides:
      - A standard forward() method that leverages the built-in nn.LSTM for efficient computation.
      - A forward_all_states() method that manually unrolls the LSTM using nn.LSTMCell modules.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input.
    hidden_size : int
        Number of features in the hidden state.
    num_layers : int
        Number of recurrent layers.
    dropout : float, optional
        If non-zero, introduces a dropout layer on the outputs of each LSTM layer except the last.
        Default is 0.0.
    batch_first : bool, optional
        If True, input and output tensors are provided as (batch, seq, feature).
        Default is True.
    """

    def __init__(
        self, input_size, hidden_size, num_layers, dropout=0.0, batch_first=True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first

        # Define dropout if needed
        if dropout > 0:
            self._dropout = nn.Dropout(dropout)

        # Built-in LSTM for training
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )

        # LSTMCell modules for manual unrolling during inference
        self.lstm_cells = nn.ModuleList()
        for layer in range(num_layers):
            curr_input_size = input_size if layer == 0 else hidden_size
            self.lstm_cells.append(nn.LSTMCell(curr_input_size, hidden_size))

        # Share parameters between the LSTM and LSTMCell modules
        self._share_weights_with_cells()
        
    def _share_weights_with_cells(self):
        """
        Share the built-in LSTM's parameters with the LSTMCell modules.
        """
        for layer in range(self.num_layers):
            cell = self.lstm_cells[layer]
            cell.weight_ih = getattr(self.lstm, f"weight_ih_l{layer}")
            cell.weight_hh = getattr(self.lstm, f"weight_hh_l{layer}")
            cell.bias_ih = getattr(self.lstm, f"bias_ih_l{layer}")
            cell.bias_hh = getattr(self.lstm, f"bias_hh_l{layer}")

    def init_hidden(self, batch_size):
        """Initialize the hidden state and cell state for the LSTM_V2 module."""
        device = next(self.parameters()).device
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        return (h0, c0)

    def forward(self, x, hx=None):
        """Standard forward pass using the built-in nn.LSTM."""
        return self.lstm(x, hx)

    def forward_all_states(self, x, hx=None, interventions=None, alphas=None):
        """
        Manually unroll the LSTM using LSTMCell modules and return outputs and all intermediate
        hidden and cell states for every timestep.
        """
        original_batch_first = self.batch_first
        if not original_batch_first:
            # Input is expected to be (timesteps, batch, input_size), convert to batch-first.
            x = x.transpose(0, 1)  # Now (batch, timesteps, input_size)
        batch_size, timesteps, _ = x.size()

        # Initialize hidden and cell states.
        if hx is None:
            h, c = self.init_hidden(batch_size)
        else:
            h, c = hx

        h_states = [
            x.new_zeros(batch_size, timesteps, self.hidden_size)
            for _ in range(self.num_layers)
        ]
        c_states = [
            x.new_zeros(batch_size, timesteps, self.hidden_size)
            for _ in range(self.num_layers)
        ]
        h_t = [h[layer] for layer in range(self.num_layers)]
        c_t = [c[layer] for layer in range(self.num_layers)]
        output = x.new_zeros(batch_size, timesteps, self.hidden_size)

        # Handle interventions if provided
        if interventions is not None:
            assert interventions.shape[-2] == len(h_t)
            assert interventions.shape[-1] == h_t[0].shape[-1]
            assert interventions.shape[1] == h_t[0].shape[0]
                        
        for t in range(timesteps):
            input_t = x[:, t, :]
            for layer in range(self.num_layers):
                if interventions is not None:
                    alpha_h = alphas[0, :, t, layer]
                    alpha_c = alphas[1, :, t, layer]
                    h_prev = (1 - alpha_h) * h_t[layer] + interventions[0, :, t, layer]
                    c_prev = (1 - alpha_c) * c_t[layer] + interventions[1, :, t, layer]
                else:
                    h_prev, c_prev = h_t[layer], c_t[layer]

                h_new, c_new = self.lstm_cells[layer](input_t, (h_prev, c_prev))
                
                # Save recurrent state
                h_t[layer] = h_new
                c_t[layer] = c_new

                # Record history
                h_states[layer][:, t, :] = h_new
                c_states[layer][:, t, :] = c_new

                # Apply dropout to feed-forward connection to next layer
                if (
                    layer < self.num_layers - 1
                    and self.dropout > 0
                    and self.training
                ):
                    input_t = self._dropout(h_new)
                else:
                    input_t = h_new                
                
            output[:, t, :] = h_t[-1]

        h_all = torch.stack(
            h_states, dim=0
        )  # (num_layers, batch, timesteps, hidden_size)
        c_all = torch.stack(
            c_states, dim=0
        )  # (num_layers, batch, timesteps, hidden_size)

        if not original_batch_first:
            # Convert back to non-batch-first format.
            output = output.transpose(0, 1)  # (timesteps, batch, hidden_size)
            h_all = h_all.transpose(
                1, 2
            )  # (num_layers, timesteps, batch, hidden_size)
            c_all = c_all.transpose(
                1, 2
            )  # (num_layers, timesteps, batch, hidden_size)
        return output, (h_all, c_all)


# ============================================================================
# SEQUENCE MODEL COMPONENTS
# ============================================================================

class SequenceModelBase(nn.Module):
    """Computational component of the sequence model.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input features.
    output_size : int
        Dimensionality of the output features.
    feedforward_size : int, optional
        Dimensionality of the hidden representation in the feedforward block (default: 32).
    recurrent_size : int, optional
        Hidden size of the recurrent module (default: 64).
    recurrent_num_layers : int, optional
        Number of layers in the recurrent module (default: 2).
    recurrent_cls : str, optional
        Class name for recurrent module (default: "LSTM_V2").
    output_mlp_size : int, optional
        Hidden size of the intermediate layer in the output MLP (default: 32).
    batch_first : bool, optional
        If True, input and output tensors are expected in (batch, timesteps, features) format
        (default: True).
    dropout : float, optional
        Dropout probability used in the feedforward, recurrent, and output modules (default: 0.0).
    output_activation : nn.Module, optional
        Activation function applied after the output MLP (default: PartialSoftmax(2)).
    feedforward_layer_norm : bool, optional
        Whether to apply layer normalization in the feedforward block (default: True).
    recurrent_layer_norm : bool, optional
        Whether to apply layer normalization in the recurrent module (default: True).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        feedforward_size: int = 32,
        recurrent_size: int = 64,
        recurrent_num_layers: int = 2,
        recurrent_cls: str = "LSTM_V2",
        output_mlp_size: int = 32,
        feedforward_kwargs: dict = None,
        recurrent_kwargs: dict = None,
        output_kwargs: dict = None,
        batch_first: bool = True,
        dropout: float = 0.0,
        output_activation: nn.Module = None,
        feedforward_layer_norm: bool = True,
        recurrent_layer_norm: bool = True,
    ):
        super().__init__()
        
        # Set all initialization parameters as instance attributes.
        self.input_size = input_size
        self.output_size = output_size
        self.feedforward_size = feedforward_size
        self.recurrent_size = recurrent_size
        self.recurrent_num_layers = recurrent_num_layers
        self.output_mlp_size = output_mlp_size
        self.feedforward_kwargs = feedforward_kwargs or {}
        self.recurrent_kwargs = recurrent_kwargs or {}
        self.output_kwargs = output_kwargs or {}
        self.batch_first = batch_first
        self.dropout = dropout
        self.output_activation = output_activation if output_activation is not None else PartialSoftmax(2)
        self.feedforward_layer_norm = feedforward_layer_norm
        self.recurrent_layer_norm = recurrent_layer_norm

        # Get the recurrent class - only support LSTM_V2 for now
        if recurrent_cls == "LSTM_V2":
            self.recurrent_cls = LSTM_V2
        else:
            raise ValueError(f"Unsupported recurrent class: {recurrent_cls}")

        # Maps from input_size -> feedforward_size.
        # Optionally applies LayerNorm and Dropout before ReLU.
        ff_layers = []
        if feedforward_size:
            ff_layers.append(
                nn.Linear(input_size, feedforward_size, **self.feedforward_kwargs)
            )
            if self.feedforward_layer_norm:
                ff_layers.append(nn.LayerNorm(feedforward_size))
            if dropout > 0:
                ff_layers.append(nn.Dropout(dropout))
            ff_layers.append(nn.ReLU())
            recurrent_input_size = feedforward_size
        else:
            ff_layers.append(nn.Identity())
            recurrent_input_size = input_size

        self.feedforward = nn.Sequential(*ff_layers)
        
        # Uses LSTM_V2: input dimension is feedforward_size; output dimension is recurrent_size over recurrent_num_layers.
        self.recurrent = self.recurrent_cls(
            input_size=recurrent_input_size,
            hidden_size=recurrent_size,
            num_layers=recurrent_num_layers,
            dropout=dropout,
            batch_first=batch_first,
            **self.recurrent_kwargs,
        )

        # Maps from recurrent_size -> output_mlp_size -> output_size, optionally with LayerNorm and Dropout.
        out_layers = []
        if self.recurrent_layer_norm:
            out_layers.append(nn.LayerNorm(recurrent_size))
        if dropout > 0:
            out_layers.append(nn.Dropout(dropout))
        if output_mlp_size:
            out_layers.extend(
                [
                    nn.Linear(recurrent_size, output_mlp_size, **self.output_kwargs),
                    nn.ReLU(),
                    nn.Linear(output_mlp_size, output_size, **self.output_kwargs),
                    self.output_activation,
                ]
            )
        else:
            out_layers.extend(
                [
                    nn.Linear(recurrent_size, output_size, **self.output_kwargs),
                    self.output_activation,
                ]
            )
        self.output = nn.Sequential(*out_layers)

    def forward(self, x, hx=None):
        """
        Forward pass through the composite model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, timesteps, input_size) if batch_first is True.
        hx : tuple, optional
            Initial hidden state and cell state for the recurrent module.

        Returns
        -------
        tuple
            A tuple (out, (h_n, c_n)) where:
              - out is the output of the model after processing through the output MLP.
              - (h_n, c_n) are the final recurrent states returned by the recurrent module.
        """
        # Pass input through feedforward block.
        x_ff = self.feedforward(x)  # -> (batch, timesteps, feedforward_size)
        # Process the sequence through the recurrent module.
        x_rec, states = self.recurrent(x_ff, hx)
        # x_rec: (batch, timesteps, recurrent_size)
        # Pass recurrent output through output MLP.
        x_out = self.output(x_rec)  # -> (batch, timesteps, output_size)
        return x_out, states

    def forward_all_states(self, x, hx=None, *args, **kwargs):
        """
        Identical to forward(), but returns all intermediate recurrent states.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, timesteps, input_size) if batch_first is True.
        hx : tuple, optional
            Initial hidden and cell states for the recurrent module.

        Returns
        -------
        tuple
            A tuple (out, (h_all, c_all)) where:
              - out is the output of the model with shape (batch, timesteps, output_size).
              - h_all and c_all contain the hidden and cell states at every timestep for each layer
                with shapes (num_layers, batch, timesteps, recurrent_size).
        """
        # Pass input through feedforward block.
        x_ff = self.feedforward(x)
        # Get the recurrent output and all intermediate states.
        x_rec, all_states = self.recurrent.forward_all_states(
            x_ff, hx, *args, **kwargs
        )
        # x_rec: (batch, timesteps, recurrent_size) if batch_first is True.
        x_out = self.output(x_rec)
        return x_out, all_states

    def detach_hiddens(self, hiddens):
        if isinstance(hiddens, tuple):
            return tuple(h.detach() for h in hiddens)
        else:
            return hiddens.detach()


class SequenceModel(LightningModule):
    """
    Simplified Lightning wrapper for SequenceModelBase for inference only.
    
    This is a minimal version focused on state extraction, removing training-specific
    functionality like optimizers, schedulers, and loss computation.
    """

    def __init__(
        self,
        model_config: dict,
        scaling: float = 255.0,
        shift: float = 0.0,
        return_all_states: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create the computational model
        self.sequence_model = SequenceModelBase(**model_config)
        
        # Model behavior settings
        self.scaling = scaling
        self.shift = shift
        self.return_all_states = return_all_states
        
        # Store config for easy access
        self.model_config = model_config

    @staticmethod
    def standardize(func):
        """Decorator to apply standardization to inputs and outputs."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            samples = (args[0] - self.shift) / self.scaling
            new_args = (samples,) + args[1:]
            outputs = func(self, *new_args, **kwargs)
            preds = outputs[0] * self.scaling + self.shift
            return (preds,) + outputs[1:]
        return wrapper

    @standardize
    def forward(self, *args, **kwargs):
        return self.sequence_model.forward(*args, **kwargs)

    @standardize
    def forward_all_states(self, *args, **kwargs):
        return self.sequence_model.forward_all_states(*args, **kwargs)

    def predict_step(self, batch, batch_idx, hiddens=None):
        """Prediction step for Lightning trainer."""
        if isinstance(batch, list):
            samples, *_ = batch
        else:
            samples = batch

        if self.return_all_states:
            return self.forward_all_states(samples, hiddens)
        else:
            return self.forward(samples, hiddens)