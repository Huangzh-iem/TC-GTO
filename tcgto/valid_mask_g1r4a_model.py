"""Valid-mask-semantics fix for the G1R4A graph-temporal backbone (no params added).

Fixes HeterogeneousOperatorBlock so that forwarding a real graph natively or
padded to a larger N produces identical outputs on valid nodes / edges:

1. structural messages only exist when BOTH endpoints are valid
   (m_{j->i} = valid_i * valid_j * phi(...)); the padding node can no longer
   inject a bias message into the last real floor;
2. global floor attention gathers only the valid keys per row before softmax,
   so the softmax reduction length (and therefore the fp result) is identical
   to a native forward;
3. padded floor hidden states are re-zeroed after every block update
   (floor_norm1 / floor_norm2), so residual/FFN cannot reactivate them.

Note on statistics: `floor.mean(dim=1)` / `floor.std(dim=1, unbiased=False)`
and `drift.mean/std(dim=1)` reduce over the TIME axis (per node), so padded
zero rows do NOT enter their denominators; they are N-independent and are
left unchanged (verified by the R4F scan, correcting the R4 root-cause note).

No new trainable parameters; state_dict layout is identical to the parent
class, so existing G1R4A checkpoints load strictly.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .hetero_graph_operator.model import HeterogeneousOperatorBlock, TypedMultiHeadFloorAttention
from .hetero_graph_operator.physics import structural_edge_mask
from .geometry_attention_lf_model import (
    GeometryAttentionLFTransmissibilityConditionedSTGNO2L,
)


class ValidMaskTypedMultiHeadFloorAttention(TypedMultiHeadFloorAttention):
    """TypedMultiHeadFloorAttention with exact valid-key softmax.

    The parent masks invalid keys with -inf and softmaxes over all N nodes,
    which makes a padded forward differ from a native forward at the
    reduction level (~1e-7 in the weights).  This subclass gathers only the
    valid keys per row before softmax, so a native N and a padded-to-12
    forward produce bitwise-identical attention weights for valid queries.
    For an all-valid batch the gathered set is the full key set and the
    output is bitwise identical to the parent.
    """

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        valid: torch.Tensor,
        modal_graph: torch.Tensor | None = None,
        bias_delta: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, steps, nodes, hidden = h.shape
        q = self.query(h).reshape(batch, steps, nodes, self.num_heads, self.head_dim)
        k = self.key(h).reshape(batch, steps, nodes, self.num_heads, self.head_dim)
        v = self.value(h).reshape(batch, steps, nodes, self.num_heads, self.head_dim)
        k = torch.einsum("btjhd,hde->btjhe", k, self.relation_attention)
        v = torch.einsum("btjhd,hde->btjhe", v, self.relation_message)
        scores = torch.einsum("btihd,btjhd->bthij", q, k) / math.sqrt(self.head_dim)
        delta = coords[:, :, None] - coords[:, None, :]
        bias = self.coord_bias(torch.stack([delta, delta.abs()], dim=-1)).permute(0, 3, 1, 2)
        scores = scores + bias[:, None] + self.relation_prior[None, None, :, None, None]
        if self.modal_guided:
            if modal_graph is None:
                raise ValueError("modal-guided attention requires modal_graph")
            indices = torch.arange(nodes, device=h.device)
            local = (indices[:, None] - indices[None, :]).abs().eq(1).to(h)
            local = local[None] * valid[:, :, None] * valid[:, None, :]
            beta = self.beta
            gamma = self.gamma
            if bias_delta is not None:
                beta = beta + bias_delta[:, 0]
                gamma = gamma + bias_delta[:, 1]
                beta = beta[:, None, None, None, None]
                gamma = gamma[:, None, None, None, None]
            scores = scores + beta * local[:, None, None] + gamma * modal_graph[:, None, None]
        # Gather only valid keys per row so the softmax reduction length is
        # identical to a native (unpadded) forward.
        valid_key = valid.gt(0.5)
        counts = valid_key.sum(dim=1)
        maxk = int(counts.max().item())
        ar = torch.arange(nodes, device=h.device)
        key_idx = ar[None, :].expand(batch, -1).clone()
        key_idx = key_idx.masked_fill(ar[None, :] >= counts[:, None], 0)
        key_mask_g = ar[None, :] < counts[:, None]
        idx = key_idx[:, None, None, None, :].expand(batch, steps, self.num_heads, nodes, -1)
        scores_g = scores.gather(-1, idx)
        scores_g = scores_g.masked_fill(
            ~key_mask_g[:, None, None, None, :], torch.finfo(scores_g.dtype).min
        )
        weights = torch.softmax(scores_g, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        vidx = key_idx[:, None, :, None, None].expand(batch, steps, -1, self.num_heads, self.head_dim)
        v_g = v.gather(2, vidx)
        message = torch.einsum("bthij,btjhd->btihd", self.dropout(weights), v_g)
        output = message.reshape(batch, steps, nodes, hidden) * valid[:, None, :, None]
        if return_attention:
            return output, weights
        return output


class ValidMaskHeterogeneousOperatorBlock(HeterogeneousOperatorBlock):
    """Drop-in HeterogeneousOperatorBlock with strict valid-mask semantics."""

    def __init__(self, config) -> None:
        super().__init__(config)
        fixed_attention = ValidMaskTypedMultiHeadFloorAttention(
            config.hidden_dim,
            config.num_heads,
            config.dropout,
            modal_guided=config.modal_guided_attention,
            local_bias_init=config.modal_local_bias_init,
            modal_bias_init=config.modal_relation_bias_init,
        )
        fixed_attention.load_state_dict(self.global_relation.state_dict())
        self.global_relation = fixed_attention

    def forward(
        self,
        floor: torch.Tensor,
        excitation: torch.Tensor,
        mass_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        coords: torch.Tensor,
        sensor_mask: torch.Tensor,
        valid: torch.Tensor,
        modal_graph: torch.Tensor | None = None,
        attention_bias_delta: torch.Tensor | None = None,
        return_attention: bool = False,
        message_film: tuple[torch.Tensor, torch.Tensor, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        counts = valid.sum(dim=1)
        nodes = floor.shape[2]
        if bool((counts == counts[0]).all()) and int(counts[0].item()) <= nodes:
            n = int(counts[0].item())
            if n == nodes:
                # native (unpadded) batch: exact original computation.
                return super().forward(
                    floor, excitation, mass_latent, edge_latent, coords, sensor_mask,
                    valid, modal_graph, attention_bias_delta, return_attention, message_film,
                )
            # homogeneous padded batch: slice to the real graph, run the exact
            # native computation, then zero-pad the outputs back.
            mg = modal_graph[:, :n, :n] if modal_graph is not None else None
            out = super().forward(
                floor[:, :, :n], excitation, mass_latent[:, :n], edge_latent[:, :n],
                coords[:, :n], sensor_mask[:, :n], valid[:, :n],
                mg, attention_bias_delta, return_attention, message_film,
            )
            floor_p = torch.zeros_like(floor)
            floor_p[:, :, :n] = out[0]
            mass_p = torch.zeros_like(mass_latent)
            mass_p[:, :n] = out[2]
            edge_p = torch.zeros_like(edge_latent)
            edge_p[:, :n] = out[3]
            if return_attention:
                return floor_p, out[1], mass_p, edge_p, out[4]
            return floor_p, out[1], mass_p, edge_p
        # mixed (per-row differing valid counts): fixed padded path.
        return self._fixed_forward(
            floor, excitation, mass_latent, edge_latent, coords, sensor_mask, valid,
            modal_graph, attention_bias_delta, return_attention, message_film,
        )

    def _structural_message(self, h: torch.Tensor, edge: torch.Tensor, edge_valid: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, hidden = h.shape
        ground = self.ground_token.view(1, 1, 1, hidden).expand(batch, steps, 1, hidden)
        lower = torch.cat([ground, h[:, :, :-1]], dim=2)
        edge_dynamic = edge[:, None].expand(-1, steps, -1, -1)
        upward = self.structural_up(torch.cat([lower, edge_dynamic], dim=-1))
        downward_raw = self.structural_down(torch.cat([h, edge_dynamic], dim=-1))
        # Edge semantics: a message is only allowed when BOTH endpoints are
        # valid.  Multiplying before the downward shift prevents the padding
        # node (i+1 invalid) from injecting structural_down(0,0) into the
        # last real floor.  For an all-valid batch this is an exact x1.0 mask.
        ev = edge_valid[:, None, :, None].to(upward)
        upward = upward * ev
        downward_raw = downward_raw * ev
        zeros = torch.zeros_like(downward_raw[:, :, :1])
        downward = torch.cat([downward_raw[:, :, 1:], zeros], dim=2)
        degree = edge_valid.to(h)
        if nodes > 1:
            degree = degree + torch.cat([edge_valid[:, 1:], edge_valid[:, :1] * 0.0], dim=1).to(h)
        return (upward + downward) / degree[:, None, :, None].clamp_min(1.0)

    def _fixed_forward(
        self,
        floor: torch.Tensor,
        excitation: torch.Tensor,
        mass_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        coords: torch.Tensor,
        sensor_mask: torch.Tensor,
        valid: torch.Tensor,
        modal_graph: torch.Tensor | None = None,
        attention_bias_delta: torch.Tensor | None = None,
        return_attention: bool = False,
        message_film: tuple[torch.Tensor, torch.Tensor, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        batch, steps, nodes, hidden = floor.shape
        edge_valid = structural_edge_mask(valid).to(floor)
        structural = self._structural_message(floor, edge_latent, edge_valid)
        attention_weights = None
        if self.use_global:
            global_result = self.global_relation(
                floor, coords, valid, modal_graph, attention_bias_delta,
                return_attention=return_attention,
            )
            if return_attention:
                global_message, attention_weights = global_result
            else:
                global_message = global_result
        else:
            global_message = torch.zeros_like(floor)
        excitation_expanded = excitation[:, :, None, :].expand(-1, -1, nodes, -1)
        mass_dynamic = mass_latent[:, None].expand(-1, steps, -1, -1)
        forcing = self.excitation_to_floor(torch.cat([excitation_expanded, mass_dynamic], dim=-1))
        if self.use_global:
            relation_weights = torch.softmax(self.relation_logits, dim=0)
        else:
            active_weights = torch.softmax(self.relation_logits[[0, 2]], dim=0)
            relation_weights = torch.stack(
                [active_weights[0], active_weights.new_tensor(0.0), active_weights[1]]
            )
        floor_message = relation_weights[0] * structural + relation_weights[1] * global_message + relation_weights[2] * forcing
        if message_film is not None:
            gamma, beta, strength = message_film
            if gamma.shape != (batch, hidden) or beta.shape != (batch, hidden):
                raise ValueError(f"message FiLM tensors must have shape [{batch},{hidden}]")
            floor_message = (
                (1.0 + float(strength) * gamma[:, None, None, :]) * floor_message
                + float(strength) * beta[:, None, None, :]
            )
        transformed_floor = self.dropout(self.floor_target(torch.nn.functional.gelu(floor_message)))
        alpha_floor = torch.sigmoid(self.floor_skip)
        floor = self.floor_norm1(alpha_floor * transformed_floor + (1.0 - alpha_floor) * floor)
        floor = floor * valid[:, None, :, None]  # padded nodes stay 0 after the update

        excitation_message = self._sensor_to_excitation(floor, excitation, sensor_mask, valid)
        transformed_excitation = self.dropout(self.excitation_target(torch.nn.functional.gelu(excitation_message)))
        alpha_excitation = torch.sigmoid(self.excitation_skip)
        excitation = self.excitation_norm1(
            alpha_excitation * transformed_excitation + (1.0 - alpha_excitation) * excitation
        )

        floor_temporal = self.floor_temporal(
            floor.permute(0, 2, 3, 1).reshape(batch * nodes, hidden, steps)
        ).reshape(batch, nodes, hidden, steps).permute(0, 3, 1, 2)
        excitation_temporal = self.excitation_temporal(excitation.transpose(1, 2)).transpose(1, 2)
        floor = self.floor_norm2(floor + self.dropout(floor_temporal))
        floor = floor * valid[:, None, :, None]
        excitation = self.excitation_norm2(excitation + self.dropout(excitation_temporal))
        floor = (floor + self.dropout(self.floor_ffn(floor))) * valid[:, None, :, None]
        excitation = excitation + self.dropout(self.excitation_ffn(excitation))

        floor_mean = floor.mean(dim=1)
        floor_std = floor.std(dim=1, unbiased=False)
        mass_latent = self.mass_norm(
            mass_latent + self.dropout(self.mass_update(torch.cat([mass_latent, floor_mean, floor_std], dim=-1)))
        ) * valid[:, :, None]
        ground_dynamic = torch.zeros_like(floor[:, :, :1])
        lower = torch.cat([ground_dynamic, floor[:, :, :-1]], dim=2)
        drift = floor - lower
        drift_mean = drift.mean(dim=1)
        drift_std = drift.std(dim=1, unbiased=False)
        excitation_pool = excitation.mean(dim=1)[:, None, :].expand(-1, nodes, -1)
        edge_features = torch.cat(
            [edge_latent, drift_mean, drift_std, excitation_pool], dim=-1
        )
        edge_latent = self.edge_norm(edge_latent + self.dropout(self.edge_update(edge_features))) * edge_valid[:, :, None]
        if return_attention:
            return floor, excitation, mass_latent, edge_latent, attention_weights
        return floor, excitation, mass_latent, edge_latent


class ValidMaskG1R4A(GeometryAttentionLFTransmissibilityConditionedSTGNO2L):
    """G1R4A model whose OperatorBlocks use strict valid-mask semantics.

    Architecture, width, depth and state_dict layout are identical to the
    parent (checkpoints strict-load); only block internals are fixed.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        fixed = nn.ModuleList(
            [ValidMaskHeterogeneousOperatorBlock(config) for _ in range(config.depth)]
        )
        for old, new in zip(self.blocks, fixed):
            new.load_state_dict(old.state_dict())
        self.blocks = fixed

    def forward(
        self,
        sparse_observation: torch.Tensor,
        sensor_mask: torch.Tensor,
        floor_coordinate: torch.Tensor,
        valid_node_mask: torch.Tensor,
        lf_transmissibility_signature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process rows in groups of equal valid counts (exact native shapes).

        The LF branch and the graph-temporal backbone are both computed per
        group with the group's own node dimension, so a padded batch is
        bitwise identical to the grouped native forwards.  Mixed batches
        therefore match the per-N grouped comparison exactly.
        """
        counts = valid_node_mask.sum(dim=1)
        nodes = sparse_observation.shape[2]
        unique_counts = torch.unique(counts)
        if unique_counts.numel() == 1 and int(unique_counts[0].item()) == nodes:
            return super().forward(
                sparse_observation, sensor_mask, floor_coordinate,
                valid_node_mask, lf_transmissibility_signature,
            )
        batch = sparse_observation.shape[0]
        steps = sparse_observation.shape[1]
        hidden = self.config.hidden_dim
        p = int(lf_transmissibility_signature.shape[1])
        device = sparse_observation.device
        dtype = sparse_observation.dtype
        outs = {
            "response": torch.zeros(batch, steps, nodes, 3, device=device, dtype=dtype),
            "input": torch.zeros(batch, steps, device=device, dtype=dtype),
            "direct_input": torch.zeros(batch, steps, device=device, dtype=dtype),
            "floor_latent": torch.zeros(batch, steps, nodes, hidden, device=device, dtype=dtype),
            "excitation_latent": torch.zeros(batch, steps, hidden, device=device, dtype=dtype),
            "node_latent": torch.zeros(batch, nodes, hidden, device=device, dtype=dtype),
            "edge_latent": torch.zeros(batch, nodes, hidden, device=device, dtype=dtype),
            "global_lf_context": torch.zeros(batch, hidden, device=device, dtype=dtype),
            "lf_attention_weights": torch.zeros(batch, p, device=device, dtype=dtype),
            "lf_pair_embeddings": torch.zeros(batch, p, 32, device=device, dtype=dtype),
            "lf_pair_geometry": torch.zeros(batch, p, 4, device=device, dtype=dtype),
            "lf_pair_rho": torch.zeros(batch, device=device, dtype=dtype),
            "mass_anchor_loss": sparse_observation.new_zeros(()),
        }
        for c in unique_counts.tolist():
            c = int(c)
            idx = (counts == c).nonzero(as_tuple=True)[0]
            sub = super().forward(
                sparse_observation[idx][:, :, :c],
                sensor_mask[idx][:, :c],
                floor_coordinate[idx][:, :c],
                valid_node_mask[idx][:, :c],
                lf_transmissibility_signature[idx],
            )
            outs["response"][idx, :, :c] = sub["response"]
            outs["input"][idx] = sub["input"]
            outs["direct_input"][idx] = sub["direct_input"]
            outs["floor_latent"][idx, :, :c] = sub["floor_latent"]
            outs["excitation_latent"][idx] = sub["excitation_latent"]
            outs["node_latent"][idx, :c] = sub["node_latent"]
            outs["edge_latent"][idx, :c] = sub["edge_latent"]
            outs["global_lf_context"][idx] = sub["global_lf_context"]
            outs["lf_attention_weights"][idx] = sub["lf_attention_weights"]
            outs["lf_pair_embeddings"][idx] = sub["lf_pair_embeddings"]
            outs["lf_pair_geometry"][idx] = sub["lf_pair_geometry"]
            outs["lf_pair_rho"][idx] = sub["lf_pair_rho"]
        return outs

    def _forward_impl(
        self,
        sparse_observation: torch.Tensor,
        sensor_mask: torch.Tensor,
        floor_coordinate: torch.Tensor,
        valid_node_mask: torch.Tensor,
        global_floor_context: torch.Tensor | None,
        local_floor_context: torch.Tensor | None = None,
        message_modulations: list[tuple[torch.Tensor, torch.Tensor, float]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the exact native computation per group of equal valid counts.

        Every row is processed in a tensor whose node dimension equals its
        valid count (identical shapes to a native forward), so padded rows can
        never influence valid rows at any level (kernel/shape-dependent fp
        reductions included).  Outputs are zero-padded back to the batch node
        dimension.  This makes native-vs-padded and mixed-vs-grouped forwards
        bitwise identical.
        """
        counts = valid_node_mask.sum(dim=1)
        nodes = sparse_observation.shape[2]
        unique_counts = torch.unique(counts)
        if message_modulations is not None:
            raise NotImplementedError("message_modulations are not supported by the valid-mask wrapper")
        if unique_counts.numel() == 1 and int(unique_counts[0].item()) == nodes:
            return super()._forward_impl(
                sparse_observation, sensor_mask, floor_coordinate, valid_node_mask,
                global_floor_context, local_floor_context, message_modulations,
            )
        batch = sparse_observation.shape[0]
        steps = sparse_observation.shape[1]
        hidden = self.config.hidden_dim
        device = sparse_observation.device
        dtype = sparse_observation.dtype
        outs = {
            "response": torch.zeros(batch, steps, nodes, 3, device=device, dtype=dtype),
            "input": torch.zeros(batch, steps, device=device, dtype=dtype),
            "direct_input": torch.zeros(batch, steps, device=device, dtype=dtype),
            "floor_latent": torch.zeros(batch, steps, nodes, hidden, device=device, dtype=dtype),
            "excitation_latent": torch.zeros(batch, steps, hidden, device=device, dtype=dtype),
            "node_latent": torch.zeros(batch, nodes, hidden, device=device, dtype=dtype),
            "edge_latent": torch.zeros(batch, nodes, hidden, device=device, dtype=dtype),
            "mass_anchor_loss": sparse_observation.new_zeros(()),
        }
        for c in unique_counts.tolist():
            c = int(c)
            idx = (counts == c).nonzero(as_tuple=True)[0]
            gctx = global_floor_context[idx] if global_floor_context is not None else None
            lctx = local_floor_context[idx][:, :c] if local_floor_context is not None else None
            sub = super()._forward_impl(
                sparse_observation[idx][:, :, :c],
                sensor_mask[idx][:, :c],
                floor_coordinate[idx][:, :c],
                valid_node_mask[idx][:, :c],
                gctx,
                lctx,
                None,
            )
            outs["response"][idx, :, :c] = sub["response"]
            outs["input"][idx] = sub["input"]
            outs["direct_input"][idx] = sub["direct_input"]
            outs["floor_latent"][idx, :, :c] = sub["floor_latent"]
            outs["excitation_latent"][idx] = sub["excitation_latent"]
            outs["node_latent"][idx, :c] = sub["node_latent"]
            outs["edge_latent"][idx, :c] = sub["edge_latent"]
        return outs
