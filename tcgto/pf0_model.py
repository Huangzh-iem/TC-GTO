"""Parameter-free STGNO-2L used only by the preregistered PF0 audit.

The forward signature is intentionally incapable of receiving structural
parameters.  It preserves the A0 message-passing/temporal backbone while
replacing the nominal-property node and edge features with graph topology.
"""
from __future__ import annotations

import torch
from torch import nn

from .hetero_graph_operator.config import HeteroGNOConfig
from .hetero_graph_operator.model import PhysicsInformedHeteroGraphOperator, _expand_coordinates
from .hetero_graph_operator.physics import structural_edge_mask


class ParameterFreeSTGNO2L(PhysicsInformedHeteroGraphOperator):
    """A0-compatible response/input operator with no M/K/C input path."""

    def __init__(self, config: HeteroGNOConfig) -> None:
        super().__init__(config)
        hidden = config.hidden_dim
        # Explicit topology encoders replace the original property encoders.
        self.topology_node_encoder = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.topology_edge_encoder = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.mass_encoder = None
        self.edge_encoder = None
        # PF0 forbids parameter prediction and modal/parameter auxiliaries.
        self.record_parameter_context = None
        self.mass_context_fusion = None
        self.edge_context_fusion = None
        self.observed_modal_encoder = None
        self.modal_floor_condition = None
        self.modal_mass_condition = None
        self.modal_edge_condition = None
        self.modal_excitation_condition = None
        self.modal_feature_encoder = None
        self.parameter_decoder = None
        self.physics_refinements = nn.ModuleList()

    def forward(
        self,
        sparse_observation: torch.Tensor,
        sensor_mask: torch.Tensor,
        floor_coordinate: torch.Tensor,
        valid_node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._forward_impl(
            sparse_observation, sensor_mask, floor_coordinate, valid_node_mask, None
        )

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
        if sparse_observation.ndim != 3:
            raise ValueError("sparse_observation must have shape [batch,time,nodes]")
        batch, steps, nodes = sparse_observation.shape
        valid = valid_node_mask.to(sparse_observation)
        sensor = sensor_mask.to(sparse_observation) * valid
        coords = _expand_coordinates(floor_coordinate, batch, nodes, sparse_observation)
        edge_valid = structural_edge_mask(valid).to(sparse_observation)
        time = torch.linspace(0.0, 1.0, steps, dtype=sparse_observation.dtype, device=sparse_observation.device)
        node_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        inverse_count = (1.0 / node_count).expand(-1, nodes)

        floor_features = torch.stack(
            [
                sparse_observation * sensor[:, None, :],
                sensor[:, None, :].expand(-1, steps, -1),
                valid[:, None, :].expand(-1, steps, -1),
                coords[:, None, :].expand(-1, steps, -1),
                time[None, :, None].expand(batch, -1, nodes),
                inverse_count[:, None, :].expand(-1, steps, -1),
            ], dim=-1,
        )
        floor = self.floor_encoder(floor_features) * valid[:, None, :, None]
        if global_floor_context is not None:
            if global_floor_context.ndim == 2:
                global_floor_context = global_floor_context[:, None, :]
            floor = (floor + global_floor_context[:, None, :, :]) * valid[:, None, :, None]
        if local_floor_context is not None:
            if local_floor_context.shape != (batch, nodes, self.config.hidden_dim):
                raise ValueError(
                    "local_floor_context must have shape "
                    f"[{batch},{nodes},{self.config.hidden_dim}]"
                )
            floor = (floor + local_floor_context[:, None, :, :]) * valid[:, None, :, None]

        lower_coord = torch.cat([torch.zeros_like(coords[:, :1]), coords[:, :-1]], dim=1)
        separation = coords - lower_coord
        lower_sensor = torch.cat([torch.zeros_like(sensor[:, :1]), sensor[:, :-1]], dim=1)
        topology_node = torch.stack([coords, valid, sensor, inverse_count], dim=-1)
        topology_edge = torch.stack(
            [lower_coord, coords, separation, edge_valid, 0.5 * (sensor + lower_sensor)], dim=-1
        )
        node_latent = self.topology_node_encoder(topology_node) * valid[:, :, None]
        edge_latent = self.topology_edge_encoder(topology_edge) * edge_valid[:, :, None]

        observed_weight = sensor / sensor.sum(dim=1, keepdim=True).clamp_min(1.0)
        observed_pool = (floor * observed_weight[:, None, :, None]).sum(dim=2)
        observed_mean = (sparse_observation * observed_weight[:, None, :]).sum(dim=2)
        observed_centered = (sparse_observation - observed_mean[:, :, None]) * sensor[:, None, :]
        observed_std = (observed_centered.square() * observed_weight[:, None, :]).sum(dim=2).clamp_min(1e-12).sqrt()
        observed_rms = (sparse_observation.square() * observed_weight[:, None, :]).sum(dim=2).clamp_min(1e-12).sqrt()
        observed_abs_mean = (sparse_observation.abs() * observed_weight[:, None, :]).sum(dim=2)
        residual_features = torch.stack([observed_mean, observed_std, observed_rms, observed_abs_mean], dim=1)
        observed_input_residual = self.observed_input_residual(residual_features).squeeze(1)
        token = self.excitation_token.view(1, 1, -1).expand(batch, steps, -1)
        excitation = self.excitation_init(torch.cat([token, observed_pool], dim=-1))

        if message_modulations is not None and len(message_modulations) != len(self.blocks):
            raise ValueError("message_modulations must provide exactly one entry per graph layer")
        for layer_index, block in enumerate(self.blocks):
            floor, excitation, node_latent, edge_latent = block(
                floor, excitation, node_latent, edge_latent, coords, sensor, valid,
                None, None, False,
                None if message_modulations is None else message_modulations[layer_index],
            )

        floor_sequence = floor.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        floor_history, _ = self.floor_history(floor_sequence)
        floor_history = self.floor_history_projection(floor_history)
        floor_history = floor_history.reshape(batch, nodes, steps, -1).permute(0, 2, 1, 3)
        floor = (floor + floor_history) * valid[:, None, :, None]
        excitation_history, _ = self.excitation_history(excitation)
        excitation = excitation + self.excitation_history_projection(excitation_history)

        response = self.response_head(floor) * valid[:, None, :, None]
        direct_input = self.input_head(excitation).squeeze(-1) + observed_input_residual
        return {
            "response": response,
            "input": direct_input,
            "direct_input": direct_input,
            "floor_latent": floor,
            "excitation_latent": excitation,
            "node_latent": node_latent,
            "edge_latent": edge_latent,
            "mass_anchor_loss": response.new_zeros(()),
        }
