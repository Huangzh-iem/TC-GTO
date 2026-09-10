"""H8-G1R0 dynamic LF-conditioned model (independent new file).

``DynamicLFTransmissibilityConditionedSTGNO2L`` subclasses the frozen
``LFTransmissibilityConditionedSTGNO2L`` without adding any parameter, so the
original H8 checkpoint state dict loads unchanged.  The forward pass derives
sensor indices from the actual ``sensor_mask``, builds pair geometry from the
real floor coordinates, and accepts any pair count P = Ns*(Ns-1).

For the canonical sensor set [0,3,7] every tensor operation is identical to
the frozen implementation (same pair order, same geometry, same encoder and
mean pooling), giving bit-compatible outputs.

The original ``ts1_lf_model.py`` is intentionally untouched.
"""
from __future__ import annotations

import torch

from .ts1_lf_model import LFTransmissibilityConditionedSTGNO2L


class DynamicLFTransmissibilityConditionedSTGNO2L(LFTransmissibilityConditionedSTGNO2L):
    def forward(
        self,
        sparse_observation: torch.Tensor,
        sensor_mask: torch.Tensor,
        floor_coordinate: torch.Tensor,
        valid_node_mask: torch.Tensor,
        lf_transmissibility_signature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if lf_transmissibility_signature.ndim != 4 or lf_transmissibility_signature.shape[-1] != 4:
            raise ValueError("expected LF signature [batch, pairs, frequency, 4]")
        batch = sparse_observation.shape[0]
        counts = sensor_mask.gt(0.5).sum(dim=1)
        if not bool((counts == counts[0]).all()):
            raise ValueError("LF branch requires a common sensor layout within a batch")
        ns = int(counts[0].detach().item())
        p = ns * (ns - 1)
        if lf_transmissibility_signature.shape[1] != p:
            raise ValueError(
                f"signature has {lf_transmissibility_signature.shape[1]} pairs but mask "
                f"implies Ns={ns} -> P={p}"
            )
        # Real sensor indices per batch element from the actual masks (no
        # hidden [0,3,7] default).  Ns must be batch-homogeneous because the
        # LF signature tensor has a fixed P = Ns*(Ns-1), but sensor positions
        # may differ per sample.
        sensor_idx_list = [torch.nonzero(sensor_mask[i].gt(0.5)).flatten() for i in range(batch)]
        if floor_coordinate.ndim == 2:
            sensor_coords = torch.stack([floor_coordinate[i][sensor_idx_list[i]] for i in range(batch)], 0)
        else:
            sensor_coords = torch.stack([floor_coordinate[sensor_idx_list[i]] for i in range(batch)], 0)
        pairs = [(i, j) for i in range(ns) for j in range(ns) if i != j]
        src_idx = torch.tensor([a for a, _ in pairs], device=sensor_coords.device, dtype=torch.long)
        dst_idx = torch.tensor([b for _, b in pairs], device=sensor_coords.device, dtype=torch.long)
        src = sensor_coords[:, src_idx]
        dst = sensor_coords[:, dst_idx]
        geometry = torch.stack([src, dst, dst - src], -1)[:, :, None, :].expand(
            -1, -1, lf_transmissibility_signature.shape[2], -1
        )
        pair_input = torch.cat([lf_transmissibility_signature.to(sparse_observation), geometry], -1)
        b, p_, f, c = pair_input.shape
        x = pair_input.reshape(b * p_, f, c).transpose(1, 2)
        pair = self.lf_pair_projection(self.lf_pair_encoder(x).squeeze(-1)).reshape(b, p_, 32)
        context = self.lf_context_projection(pair.mean(1))
        output = self._forward_impl(
            sparse_observation, sensor_mask, floor_coordinate, valid_node_mask, context
        )
        output["global_lf_context"] = context
        return output
