from __future__ import annotations

import copy

import torch
from torch import nn

from .model import PhysicsInformedHeteroGraphOperator


HIGH_FREQUENCY_CUTOFF_HZ = 4.0


def split_frequency_bands(
    signal: torch.Tensor,
    dt: float,
    cutoff_hz: float = HIGH_FREQUENCY_CUTOFF_HZ,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact fixed FFT decomposition used by MR0 training and audit."""
    if signal.ndim != 2:
        raise ValueError("signal must have shape [batch,time]")
    frequencies = torch.fft.rfftfreq(signal.shape[1], d=float(dt), device=signal.device)
    spectrum = torch.fft.rfft(signal, dim=1)
    low_mask = (frequencies < float(cutoff_hz)).to(spectrum.dtype)
    low = torch.fft.irfft(spectrum * low_mask[None], n=signal.shape[1], dim=1)
    return low, signal - low


class MultiScaleHighFrequencyDecoder(nn.Module):
    """One fixed set of lightweight dilated temporal receptive fields."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        branch_hidden = max(hidden_dim // 4, 16)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Conv1d(hidden_dim, branch_hidden, kernel_size=1)
        self.branches = nn.ModuleList([
            nn.Conv1d(branch_hidden, branch_hidden, kernel_size=3, padding=dilation, dilation=dilation)
            for dilation in (1, 2, 4)
        ])
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(3 * branch_hidden, branch_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(branch_hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, excitation_latent: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(self.input_norm(excitation_latent).transpose(1, 2))
        features = torch.cat([torch.nn.functional.gelu(branch(value)) for branch in self.branches], dim=1)
        return self.output(features).squeeze(1)


class MultiResolutionInputOperator(nn.Module):
    """Frozen v7 operator with a low/high multiresolution input decoder."""

    def __init__(self, base: PhysicsInformedHeteroGraphOperator) -> None:
        super().__init__()
        self.base = base
        self.config = base.config
        self.low_head = copy.deepcopy(base.input_head)
        self.high_head = MultiScaleHighFrequencyDecoder(base.config.hidden_dim)

    def set_stage(self, stage: int) -> None:
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.low_head.parameters():
            parameter.requires_grad_(True)
        for parameter in self.high_head.parameters():
            parameter.requires_grad_(True)
        if stage == 2:
            for parameter in self.base.blocks[-1].parameters():
                parameter.requires_grad_(True)

    def set_training_modes(self, stage: int) -> None:
        self.base.eval()
        self.low_head.train()
        self.high_head.train()
        if stage == 2:
            self.base.blocks[-1].train()
            # cuDNN requires recurrent modules to be in training mode when a
            # gradient is propagated through them to an earlier trainable
            # block. Their parameters remain frozen and the one-layer GRUs
            # have no recurrent dropout, so this changes no model capacity.
            self.base.floor_history.train()
            self.base.excitation_history.train()

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        output = self.base(*args, **kwargs)
        latent = output["excitation_latent"]
        p_low = self.low_head(latent).squeeze(-1)
        p_high = self.high_head(latent)
        output["base_input"] = output["input"]
        output["p_low"] = p_low
        output["p_high"] = p_high
        output["input"] = p_low + p_high
        return output
