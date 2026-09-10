"""H8-G1R4A geometry-aware LF attention model (independent new file).

Subclasses ``DynamicLFTransmissibilityConditionedSTGNO2L`` and replaces ONLY
the LF pair aggregation:

    mean_pool(e_ij)  ->  sum_ij alpha_ij * e_ij,
    alpha = softmax(psi(e_ij, [z_i, z_j, dz, |dz|], Ns/N))

The attention scorer is a tiny MLP (37 -> 32 -> 1) whose final layer is
zero-initialized, so at initialization alpha = 1/P and the pooled context is
exactly the mean-pooling output (numerical equivalence is verified by the
pilot audit).  No other module changes; the graph-temporal backbone, heads
and residual are untouched.
"""
from __future__ import annotations

import torch
from torch import nn

from .dynamic_lf_model import (
    DynamicLFTransmissibilityConditionedSTGNO2L,
)


class GeometryAttentionLFTransmissibilityConditionedSTGNO2L(
    DynamicLFTransmissibilityConditionedSTGNO2L
):
    def __init__(self, config) -> None:
        super().__init__(config)
        # d=32 pair embedding + 4 geometry channels + 1 cardinality ratio = 37.
        self.lf_attention_scorer = nn.Sequential(
            nn.Linear(32 + 4 + 1, 32), nn.GELU(), nn.Linear(32, 1)
        )
        nn.init.zeros_(self.lf_attention_scorer[-1].weight)
        nn.init.zeros_(self.lf_attention_scorer[-1].bias)

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
            raise ValueError("LF branch requires a common sensor count within a batch")
        ns = int(counts[0].detach().item())
        p = ns * (ns - 1)
        if lf_transmissibility_signature.shape[1] != p:
            raise ValueError(
                f"signature has {lf_transmissibility_signature.shape[1]} pairs but mask "
                f"implies Ns={ns} -> P={p}"
            )
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

        # Geometry-aware set attention over the P directed pairs.
        g = torch.stack([src, dst, dst - src, (dst - src).abs()], -1)  # [B,P,4]
        valid_count = valid_node_mask.gt(0.5).sum(dim=1).clamp_min(1.0).to(sparse_observation)
        rho = (torch.as_tensor(ns, dtype=sparse_observation.dtype, device=sparse_observation.device) / valid_count)  # [B]
        score_input = torch.cat(
            [pair, g, rho[:, None, None].expand(-1, p_, 1)], -1
        )  # [B,P,37]
        logits = self.lf_attention_scorer(score_input).squeeze(-1)  # [B,P]
        alpha = torch.softmax(logits, dim=1)
        pooled = (alpha.unsqueeze(-1) * pair).sum(dim=1)  # [B,32]
        context = self.lf_context_projection(pooled)

        output = self._forward_impl(
            sparse_observation, sensor_mask, floor_coordinate, valid_node_mask, context
        )
        output["global_lf_context"] = context
        output["lf_attention_weights"] = alpha
        output["lf_pair_embeddings"] = pair
        output["lf_pair_geometry"] = g
        output["lf_pair_rho"] = rho
        return output
