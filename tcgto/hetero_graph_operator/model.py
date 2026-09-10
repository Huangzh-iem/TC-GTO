from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import HeteroGNOConfig
from .physics import assemble_shear_mck, mck_dynamic_residual, mck_modal_properties, structural_edge_mask


def _masked_standardize_log(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_values = torch.log(values.clamp_min(1.0e-12))
    weight = mask.to(log_values)
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (log_values * weight).sum(dim=1, keepdim=True) / count
    variance = ((log_values - mean).square() * weight).sum(dim=1, keepdim=True) / count
    return ((log_values - mean) / torch.sqrt(variance + 1.0e-6)) * weight


def _expand_coordinates(coords: torch.Tensor, batch: int, nodes: int, reference: torch.Tensor) -> torch.Tensor:
    coords = coords.to(device=reference.device, dtype=reference.dtype)
    if coords.ndim == 1:
        coords = coords.unsqueeze(0).expand(batch, -1)
    if coords.shape != (batch, nodes):
        raise ValueError(f"coords must have shape [{nodes}] or [{batch},{nodes}]")
    return coords


def modal_relation_graph(
    nominal_mass: torch.Tensor,
    nominal_stiffness: torch.Tensor,
    nominal_damping: torch.Tensor,
    valid_node_mask: torch.Tensor,
    num_modes: int = 3,
) -> torch.Tensor:
    """Dense modal-relation prior from nominal low-order mode shapes.

    The graph uses only nominal structure information already available to the
    v7 model.  Each mode is normalized over valid floors, modes receive equal
    weight, and cosine similarity is mapped from [-1, 1] to [0, 1].
    """

    modes = mck_modal_properties(
        nominal_mass, nominal_stiffness, nominal_damping, valid_node_mask, num_modes
    )
    shape = modes["shape"]
    valid = valid_node_mask.to(shape)
    mode_valid = modes["valid"].to(shape)
    shape = shape * valid[:, :, None] * mode_valid[:, None, :]
    mode_norm = torch.sqrt((shape.square() * valid[:, :, None]).sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    signature = shape / mode_norm
    signature = signature / math.sqrt(float(max(num_modes, 1)))
    node_norm = torch.linalg.vector_norm(signature, dim=-1, keepdim=True).clamp_min(1.0e-12)
    unit = signature / node_norm
    similarity = torch.einsum("bir,bjr->bij", unit, unit)
    pair_valid = valid[:, :, None] * valid[:, None, :]
    return (0.5 * (similarity.clamp(-1.0, 1.0) + 1.0)) * pair_valid


def layout_observability_descriptor(
    sensor_mask: torch.Tensor,
    coords: torch.Tensor,
    nominal_mass: torch.Tensor,
    nominal_stiffness: torch.Tensor,
    nominal_damping: torch.Tensor,
    valid_node_mask: torch.Tensor,
    num_modes: int = 3,
) -> torch.Tensor:
    """Fixed-width layout descriptor for mixed sensor counts.

    The first three entries are min/median/max observed heights.  For three
    sensors this is exactly the historical sorted-coordinate descriptor.
    """
    observed = (sensor_mask > 0.5) & (valid_node_mask > 0.5)
    counts = observed.sum(dim=1)
    if bool((counts < 1).any()):
        raise ValueError("layout conditioning requires at least one observed sensor")
    sorted_coords = coords.masked_fill(~observed, torch.inf).sort(dim=1).values
    batch_index = torch.arange(coords.shape[0], device=coords.device)
    lower_index = ((counts - 1) // 2).to(torch.long)
    upper_index = (counts // 2).to(torch.long)
    minimum = sorted_coords[:, 0]
    median = 0.5 * (sorted_coords[batch_index, lower_index] + sorted_coords[batch_index, upper_index])
    maximum = sorted_coords[batch_index, (counts - 1).to(torch.long)]
    position_summary = torch.stack([minimum, median, maximum], dim=1)
    modes = mck_modal_properties(
        nominal_mass, nominal_stiffness, nominal_damping, valid_node_mask, num_modes
    )
    shape = modes["shape"][:, :, :num_modes]
    shape = shape / torch.linalg.vector_norm(shape, dim=1, keepdim=True).clamp_min(1.0e-12)
    selected = shape * observed[:, :, None].to(shape)
    gram = selected.transpose(1, 2) @ selected
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    sigma_min = eigenvalues[:, 0].sqrt()
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)[None]
    logdet = torch.linalg.slogdet(gram + 1.0e-6 * eye).logabsdet
    span = maximum - minimum
    return torch.cat([position_summary, span[:, None], sigma_min[:, None], logdet[:, None]], dim=1)


def sensor_density_descriptor(
    sensor_mask: torch.Tensor,
    coords: torch.Tensor,
    nominal_mass: torch.Tensor,
    nominal_stiffness: torch.Tensor,
    nominal_damping: torch.Tensor,
    valid_node_mask: torch.Tensor,
    num_modes: int = 3,
) -> torch.Tensor:
    """Return [sensor density, normalized height span, modal coverage]."""
    observed = (sensor_mask > 0.5) & (valid_node_mask > 0.5)
    count = observed.sum(dim=1).to(coords)
    valid_count = valid_node_mask.gt(0.5).sum(dim=1).to(coords).clamp_min(1.0)
    minimum = coords.masked_fill(~observed, torch.inf).min(dim=1).values
    maximum = coords.masked_fill(~observed, -torch.inf).max(dim=1).values
    modes = mck_modal_properties(
        nominal_mass, nominal_stiffness, nominal_damping, valid_node_mask, num_modes
    )
    shape = modes["shape"][:, :, :num_modes]
    shape = shape / torch.linalg.vector_norm(shape, dim=1, keepdim=True).clamp_min(1.0e-12)
    selected = shape * observed[:, :, None].to(shape)
    coverage = torch.linalg.matrix_norm(selected, dim=(-2, -1)) / math.sqrt(float(num_modes))
    return torch.stack([count / valid_count, maximum - minimum, coverage], dim=1)


class LayoutAttentionConditioner(nn.Module):
    """Small zero-start layout hypernetwork for per-layer modal/local biases."""

    def __init__(self, depth: int, hidden_dim: int) -> None:
        super().__init__()
        self.depth = int(depth)
        self.network = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * depth)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.network(descriptor).reshape(descriptor.shape[0], self.depth, 2)


class SensorDensityConditioner(nn.Module):
    """Zero-start density/span/modal-coverage modulation of per-layer biases."""

    def __init__(self, depth: int, hidden_dim: int) -> None:
        super().__init__()
        self.depth = int(depth)
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * depth)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.network(descriptor).reshape(descriptor.shape[0], self.depth, 2)


class LayoutFeatureFiLM(nn.Module):
    """Optional zero-start layout modulation for the initial floor features."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * feature_dim)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, floor: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        scale, shift = self.network(descriptor).chunk(2, dim=-1)
        return floor * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]


class TypedMultiHeadFloorAttention(nn.Module):
    """Floor-to-floor global operator relation with coordinate bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        *,
        modal_guided: bool = False,
        local_bias_init: float = 0.05,
        modal_bias_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relation_attention = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        self.relation_message = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        self.relation_prior = nn.Parameter(torch.ones(num_heads))
        self.coord_bias = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_heads))
        self.modal_guided = bool(modal_guided)
        if self.modal_guided:
            self.beta = nn.Parameter(torch.tensor(float(local_bias_init)))
            self.gamma = nn.Parameter(torch.tensor(float(modal_bias_init)))
        else:
            self.register_parameter("beta", None)
            self.register_parameter("gamma", None)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_attention)
        nn.init.xavier_uniform_(self.relation_message)

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
        key_mask = valid[:, None, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        message = torch.einsum("bthij,btjhd->btihd", self.dropout(weights), v)
        output = message.reshape(batch, steps, nodes, hidden) * valid[:, None, :, None]
        if return_attention:
            return output, weights
        return output


class ObservationSetConditioner(nn.Module):
    """Permutation-invariant observation-set to structural-node cross attention."""

    def __init__(self, hidden_dim: int, distance_tau: float = 0.25) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.distance_tau = float(distance_tau)
        self.measurement_encoder = nn.Linear(1, hidden_dim)
        self.position_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.beta_distance = nn.Parameter(torch.tensor(0.10))
        self.gamma_modal = nn.Parameter(torch.tensor(0.10))
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def sensor_indices(sensor_mask: torch.Tensor) -> torch.Tensor:
        counts = sensor_mask.gt(0.5).sum(dim=1)
        if not bool((counts == counts[0]).all()):
            raise ValueError("observation-set batching requires a common sensor count")
        count = int(counts[0].detach().item())
        if count < 1:
            raise ValueError("observation set cannot be empty")
        return torch.stack([
            torch.nonzero(row > 0.5, as_tuple=False).flatten() for row in sensor_mask
        ], dim=0).reshape(sensor_mask.shape[0], count)

    def encode_tokens(self, signals: torch.Tensor, sensor_coords: torch.Tensor) -> torch.Tensor:
        return self.measurement_encoder(signals[..., None]) + self.position_encoder(
            sensor_coords[:, None, :, None]
        )

    def condition_from_set(
        self,
        floor: torch.Tensor,
        sensor_tokens: torch.Tensor,
        node_coords: torch.Tensor,
        sensor_coords: torch.Tensor,
        sensor_indices: torch.Tensor,
        modal_graph: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.query(floor)
        k = self.key(sensor_tokens)
        v = self.value(sensor_tokens)
        scores = torch.einsum("btih,btsh->btis", q, k) / math.sqrt(self.hidden_dim)
        distance = -(node_coords[:, :, None] - sensor_coords[:, None, :]).abs() / self.distance_tau
        modal = torch.gather(
            modal_graph,
            2,
            sensor_indices[:, None, :].expand(-1, modal_graph.shape[1], -1),
        )
        scores = scores + self.beta_distance * distance[:, None] + self.gamma_modal * modal[:, None]
        scores = scores.masked_fill(~valid[:, None, :, None].gt(0.5), torch.finfo(scores.dtype).min)
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        message = torch.einsum("btis,btsh->btih", weights, v)
        conditioned = self.norm(floor + self.output(message)) * valid[:, None, :, None]
        excitation_pool = sensor_tokens.mean(dim=2)
        return conditioned, excitation_pool, weights

    def forward(
        self,
        floor: torch.Tensor,
        sparse_acceleration: torch.Tensor,
        sensor_mask: torch.Tensor,
        coords: torch.Tensor,
        modal_graph: torch.Tensor,
        valid: torch.Tensor,
        preencoded_nodes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.sensor_indices(sensor_mask)
        sensor_coords = torch.gather(coords, 1, indices)
        if preencoded_nodes is None:
            signals = torch.gather(
                sparse_acceleration, 2, indices[:, None, :].expand(-1, sparse_acceleration.shape[1], -1)
            )
            tokens = self.encode_tokens(signals, sensor_coords)
        else:
            tokens = torch.gather(
                preencoded_nodes,
                2,
                indices[:, None, :, None].expand(-1, preencoded_nodes.shape[1], -1, preencoded_nodes.shape[-1]),
            ) + self.position_encoder(sensor_coords[:, None, :, None])
        return self.condition_from_set(
            floor, tokens, coords, sensor_coords, indices, modal_graph, valid
        )


class HeterogeneousOperatorBlock(nn.Module):
    """One typed operator block inspired by HGT's meta-relation transforms.

    Relations are deliberately separate:
      structural_coupling: adjacent physical floor edges carrying k/c latent;
      global_operator: learned nonlocal floor interaction (information only);
      sensor_to_excitation: observed floors update the input node;
      excitation_to_floor: the input node sends forcing information to floors.
    """

    def __init__(self, config: HeteroGNOConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.use_global = config.use_global_operator_edges
        self.global_relation = TypedMultiHeadFloorAttention(
            hidden,
            config.num_heads,
            config.dropout,
            modal_guided=config.modal_guided_attention,
            local_bias_init=config.modal_local_bias_init,
            modal_bias_init=config.modal_relation_bias_init,
        )
        self.ground_token = nn.Parameter(torch.zeros(hidden))
        self.structural_up = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.structural_down = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.excitation_to_floor = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.floor_to_excitation_key = nn.Linear(hidden, hidden)
        self.floor_to_excitation_value = nn.Linear(hidden, hidden)
        self.excitation_query = nn.Linear(hidden, hidden)
        self.excitation_message = nn.Linear(hidden, hidden)
        self.relation_logits = nn.Parameter(torch.zeros(3))

        self.floor_target = nn.Linear(hidden, hidden)
        self.excitation_target = nn.Linear(hidden, hidden)
        self.floor_skip = nn.Parameter(torch.tensor(1.0))
        self.excitation_skip = nn.Parameter(torch.tensor(1.0))
        self.floor_norm1 = nn.LayerNorm(hidden)
        self.floor_norm2 = nn.LayerNorm(hidden)
        self.excitation_norm1 = nn.LayerNorm(hidden)
        self.excitation_norm2 = nn.LayerNorm(hidden)
        padding = config.temporal_kernel_size // 2
        self.floor_temporal = nn.Conv1d(hidden, hidden, config.temporal_kernel_size, padding=padding)
        self.excitation_temporal = nn.Conv1d(hidden, hidden, config.temporal_kernel_size, padding=padding)
        self.floor_ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(2 * hidden, hidden))
        self.excitation_ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(2 * hidden, hidden))
        self.mass_update = nn.Sequential(nn.LayerNorm(3 * hidden), nn.Linear(3 * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.edge_update = nn.Sequential(nn.LayerNorm(4 * hidden), nn.Linear(4 * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.mass_norm = nn.LayerNorm(hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(config.dropout)

    def _structural_message(self, h: torch.Tensor, edge: torch.Tensor, edge_valid: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, hidden = h.shape
        ground = self.ground_token.view(1, 1, 1, hidden).expand(batch, steps, 1, hidden)
        lower = torch.cat([ground, h[:, :, :-1]], dim=2)
        edge_dynamic = edge[:, None].expand(-1, steps, -1, -1)
        upward = self.structural_up(torch.cat([lower, edge_dynamic], dim=-1))
        downward_raw = self.structural_down(torch.cat([h, edge_dynamic], dim=-1))
        zeros = torch.zeros_like(downward_raw[:, :, :1])
        downward = torch.cat([downward_raw[:, :, 1:], zeros], dim=2)
        degree = edge_valid.to(h)
        if nodes > 1:
            degree = degree + torch.cat([edge_valid[:, 1:], edge_valid[:, :1] * 0.0], dim=1).to(h)
        return (upward + downward) / degree[:, None, :, None].clamp_min(1.0)

    def _physical_scale_structural_message(
        self,
        h: torch.Tensor,
        edge: torch.Tensor,
        coords: torch.Tensor,
        valid: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Parameter-free, coordinate-scale normalized local aggregation.

        ``structural_up`` and ``structural_down`` remain the trained directed
        message maps.  Only the relation set and its normalized quadrature
        weights change.  The virtual ground relation is retained at floor zero
        so the original chain boundary semantics are not discarded.
        """
        batch, steps, nodes, hidden = h.shape
        coords = _expand_coordinates(coords, batch, nodes, h)
        target = coords[:, :, None]
        source = coords[:, None, :]
        distance = (source - target).abs()
        valid_pairs = valid[:, :, None].bool() & valid[:, None, :].bool()
        identity = torch.eye(nodes, device=h.device, dtype=torch.bool)[None]
        candidate = (distance > 0.0) & (distance < float(radius)) & valid_pairs & ~identity

        # Preserve every real chain edge if a nonuniform geometry puts one
        # beyond r0.  The normal G2R5C geometries do not require this union,
        # but the condition makes the guarantee explicit.
        index = torch.arange(nodes, device=h.device)
        physical = (index[:, None] - index[None, :]).abs().eq(1)[None] & valid_pairs
        candidate = candidate | physical
        raw_weight = (1.0 - distance / float(radius)).clamp_min(0.0) * candidate.to(h)
        raw_weight = torch.where(physical & raw_weight.eq(0.0), torch.ones_like(raw_weight), raw_weight)

        edge_dynamic = edge[:, None].expand(-1, steps, -1, -1)
        # Evaluate only support-set relations, rather than materialising an
        # [B,T,N,N,H] tensor and masking it afterwards.  This is algebraically
        # identical and particularly important because N=8 training has only
        # nearest-neighbour candidates at r0=2/7.
        messages = []
        for target_index in range(nodes):
            active = torch.nonzero(candidate[:, target_index].any(dim=0), as_tuple=False).flatten()
            numerator = h.new_zeros(batch, steps, hidden)
            if active.numel():
                lower = active[active < target_index]
                upper = active[active > target_index]
                if lower.numel():
                    source_state = h[:, :, lower]
                    target_edge = edge_dynamic[:, :, target_index, None].expand(-1, -1, lower.numel(), -1)
                    value = self.structural_up(torch.cat([source_state, target_edge], dim=-1))
                    weight = raw_weight[:, target_index, lower]
                    numerator = numerator + (value * weight[:, None, :, None]).sum(dim=2)
                if upper.numel():
                    source_state = h[:, :, upper]
                    source_edge = edge_dynamic[:, :, upper]
                    value = self.structural_down(torch.cat([source_state, source_edge], dim=-1))
                    weight = raw_weight[:, target_index, upper]
                    numerator = numerator + (value * weight[:, None, :, None]).sum(dim=2)
            if target_index == 0:
                # Story zero connects a virtual ground token to floor zero. It
                # is an existing physical relation, not a self-loop.
                ground = self.ground_token.view(1, 1, hidden).expand(batch, steps, -1)
                ground_message = self.structural_up(torch.cat([ground, edge_dynamic[:, :, 0]], dim=-1))
                ground_weight = raw_weight[:, 0, 1] if nodes > 1 else valid[:, 0].to(h)
                numerator = numerator + ground_message * ground_weight[:, None, None]
                denominator = raw_weight[:, 0].sum(dim=1) + ground_weight
            else:
                denominator = raw_weight[:, target_index].sum(dim=1)
            messages.append(numerator / denominator[:, None, None].clamp_min(1.0e-12))
        return torch.stack(messages, dim=2)

    def _sensor_to_excitation(
        self, h: torch.Tensor, excitation: torch.Tensor, sensor_mask: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        observed = (sensor_mask > 0.5) & (valid > 0.5)
        observed = torch.where(observed.any(dim=1, keepdim=True), observed, valid > 0.5)
        q = self.excitation_query(excitation)
        k = self.floor_to_excitation_key(h)
        v = self.floor_to_excitation_value(h)
        scores = torch.einsum("bth,btih->bti", q, k) / math.sqrt(h.shape[-1])
        scores = scores.masked_fill(~observed[:, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.einsum("bti,btih->bth", weights, v)
        return self.excitation_message(pooled)

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
        batch, steps, nodes, hidden = floor.shape
        edge_valid = structural_edge_mask(valid).to(floor)
        radius = getattr(self, "physical_scale_radius", None)
        structural = (
            self._physical_scale_structural_message(floor, edge_latent, coords, valid, float(radius))
            if radius is not None else self._structural_message(floor, edge_latent, edge_valid)
        )
        attention_weights = None
        if self.use_global:
            global_result = self.global_relation(
                floor, coords, valid, modal_graph, attention_bias_delta,
                return_attention=return_attention
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
        excitation_pool = excitation.mean(dim=1)[:, None, :].expand(-1, nodes, -1)
        edge_features = torch.cat(
            [edge_latent, drift.mean(dim=1), drift.std(dim=1, unbiased=False), excitation_pool], dim=-1
        )
        edge_latent = self.edge_norm(edge_latent + self.dropout(self.edge_update(edge_features))) * edge_valid[:, :, None]
        if return_attention:
            return floor, excitation, mass_latent, edge_latent, attention_weights
        return floor, excitation, mass_latent, edge_latent


class BoundedMCKDecoder(nn.Module):
    """KP-DEIG-style bounded log correction with optional total-mass anchor."""

    def __init__(self, hidden_dim: int, config: HeteroGNOConfig) -> None:
        super().__init__()
        self.config = config
        self.mass_head = nn.Linear(hidden_dim, 1)
        self.edge_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        mass_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        nominal_mass: torch.Tensor,
        nominal_stiffness: torch.Tensor,
        nominal_damping: torch.Tensor,
        valid: torch.Tensor,
        total_mass_prior: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        raw_mass = self.mass_head(mass_latent).squeeze(-1)
        raw_edge = self.edge_head(edge_latent)
        log_mass = self.config.mass_log_bound * torch.tanh(raw_mass)
        log_stiffness = self.config.stiffness_log_bound * torch.tanh(raw_edge[..., 0])
        log_damping = self.config.damping_log_bound * torch.tanh(raw_edge[..., 1])
        edge_valid = structural_edge_mask(valid).to(nominal_mass)
        mass = nominal_mass * torch.exp(log_mass) * valid
        stiffness = nominal_stiffness * torch.exp(log_stiffness) * edge_valid
        damping = nominal_damping * torch.exp(log_damping) * edge_valid

        anchor = self.config.mass_anchor
        if anchor.enabled:
            if total_mass_prior is None:
                if anchor.value is not None:
                    total_mass_prior = mass.new_full((mass.shape[0],), float(anchor.value))
                else:
                    total_mass_prior = (nominal_mass * valid).sum(dim=1)
            total_mass_prior = total_mass_prior.to(mass).reshape(-1)
            predicted_total = mass.sum(dim=1).clamp_min(1.0e-12)
            anchor_loss = ((predicted_total - total_mass_prior) / total_mass_prior.clamp_min(1.0e-12)).square().mean()
            if anchor.enforcement == "exact":
                mass = mass * (total_mass_prior / predicted_total)[:, None]
                anchor_loss = mass.new_tensor(0.0)
        else:
            anchor_loss = mass.new_tensor(0.0)

        return {
            "mass": mass,
            "stiffness": stiffness,
            "damping": damping,
            "log_mass_correction": log_mass * valid,
            "log_stiffness_correction": log_stiffness * edge_valid,
            "log_damping_correction": log_damping * edge_valid,
            "mass_anchor_loss": anchor_loss,
        }


class RecordParameterContext(nn.Module):
    """Encode record-scale evidence for time-invariant M/C/K graph attributes."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        self.mass_projection = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim), nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU()
        )
        self.edge_projection = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim), nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU()
        )

    def forward(
        self,
        sparse_acceleration: torch.Tensor,
        sensor_mask: torch.Tensor,
        valid: torch.Tensor,
        coords: torch.Tensor,
        mass_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, nodes = sparse_acceleration.shape
        time = torch.linspace(0.0, 1.0, steps, dtype=sparse_acceleration.dtype, device=sparse_acceleration.device)
        features = torch.stack(
            [
                sparse_acceleration,
                sensor_mask[:, None, :].expand(-1, steps, -1),
                valid[:, None, :].expand(-1, steps, -1),
                coords[:, None, :].expand(-1, steps, -1),
                time[None, :, None].expand(batch, -1, nodes),
                mass_log[:, None, :].expand(-1, steps, -1),
            ],
            dim=-1,
        )
        context = self.encoder(features) * valid[:, None, :, None]
        hidden = context.shape[-1]
        temporal = self.temporal(context.permute(0, 2, 3, 1).reshape(batch * nodes, hidden, steps))
        context = context + torch.nn.functional.gelu(
            temporal.reshape(batch, nodes, hidden, steps).permute(0, 3, 1, 2)
        )
        context = context * valid[:, None, :, None]
        mean = context.mean(dim=1)
        std = context.std(dim=1, unbiased=False)
        drift = context[:, -1] - context[:, 0]
        mass_context = self.mass_projection(torch.cat([mean, std, drift], dim=-1)) * valid[:, :, None]
        lower = torch.cat([torch.zeros_like(mass_context[:, :1]), mass_context[:, :-1]], dim=1)
        edge_context = self.edge_projection(
            torch.cat([mass_context, lower, mass_context - lower, mean], dim=-1)
        )
        edge_context = edge_context * structural_edge_mask(valid)[:, :, None].to(edge_context)
        return mass_context, edge_context


class ObservedModalEvidenceEncoder(nn.Module):
    """Extract a record-scale modal token from sparse observed accelerations.

    FFT features are formed only from observed response channels and masks.  No
    predicted or true M/C/K is used by this branch at inference.
    """

    def __init__(self, hidden_dim: int, num_modes: int) -> None:
        super().__init__()
        spectral_hidden = max(hidden_dim // 2, 16)
        self.num_modes = num_modes
        self.spectral = nn.Sequential(
            nn.Conv1d(3, spectral_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(spectral_hidden, spectral_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.sensor_projection = nn.Sequential(
            nn.Linear(spectral_hidden * 32 + 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.sensor_score = nn.Linear(hidden_dim, 1)
        self.frequency_head = nn.Linear(hidden_dim, num_modes)
        self.damping_head = nn.Linear(hidden_dim, num_modes)
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_modes)
        )

    def forward(
        self, modal_context: torch.Tensor, sensor_mask: torch.Tensor, valid: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, nodes = modal_context.shape
        spectrum = torch.fft.rfft(modal_context, dim=1).permute(0, 2, 1)
        amplitude = spectrum.abs().clamp_min(1.0e-8)
        features = torch.stack(
            [torch.log1p(amplitude), spectrum.real / amplitude, spectrum.imag / amplitude], dim=2
        ).reshape(batch * nodes, 3, -1)
        spectral = self.spectral(features).flatten(1).reshape(batch, nodes, -1)
        sensor_features = torch.cat([spectral, coords[:, :, None], sensor_mask[:, :, None]], dim=-1)
        sensor_tokens = self.sensor_projection(sensor_features)
        observed = (sensor_mask > 0.5) & (valid > 0.5)
        observed = torch.where(observed.any(dim=1, keepdim=True), observed, valid > 0.5)
        scores = self.sensor_score(sensor_tokens).squeeze(-1).masked_fill(
            ~observed, torch.finfo(sensor_tokens.dtype).min
        )
        weights = torch.softmax(scores, dim=1)
        token = torch.einsum("bn,bnh->bh", torch.nan_to_num(weights), sensor_tokens)
        log_frequency = self.frequency_head(token)
        damping = 0.20 * torch.sigmoid(self.damping_head(token))
        shape_input = torch.cat(
            [token[:, None, :].expand(-1, nodes, -1), coords[:, :, None], valid[:, :, None]], dim=-1
        )
        shape = self.shape_head(shape_input) * valid[:, :, None]
        return token, log_frequency, damping, shape


class PhysicsGraphCorrection(nn.Module):
    """One differentiable residual-driven update over all hetero graph types."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.floor_update = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_update = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.mass_update = nn.Sequential(nn.Linear(hidden_dim + 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_update = nn.Sequential(nn.Linear(hidden_dim + 7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.floor_norm = nn.LayerNorm(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.mass_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, floor: torch.Tensor, excitation: torch.Tensor, mass_latent: torch.Tensor, edge_latent: torch.Tensor,
        response: torch.Tensor, input_acceleration: torch.Tensor, mass: torch.Tensor, stiffness: torch.Tensor,
        damping: torch.Tensor, valid: torch.Tensor, dt: float, normalizer: dict[str, object],
        observed_modal: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, *_ = mck_dynamic_residual(
            response, input_acceleration, mass, stiffness, damping, valid, dt, normalizer
        )
        acceleration_scale = max(float(normalizer["response_std"][2]), float(normalizer["input_std"]), 1.0e-8)
        residual = residual / (mass[:, None, :] * acceleration_scale).clamp_min(1.0e-8)
        residual = torch.cat([torch.zeros_like(residual[:, :1]), residual, torch.zeros_like(residual[:, :1])], dim=1)
        floor = self.floor_norm(floor + self.floor_update(torch.cat([floor, residual[:, :, :, None]], dim=-1)))
        floor = floor * valid[:, None, :, None]
        global_residual = residual.abs().mean(dim=2, keepdim=True)
        excitation = self.input_norm(excitation + self.input_update(torch.cat([excitation, global_residual], dim=-1)))
        node_residual = residual.abs().mean(dim=1)
        modal_global = mass.new_zeros(mass.shape[0], 3)
        node_modal = torch.zeros_like(mass)
        if observed_modal is not None:
            modes = observed_modal["log_frequency"].shape[1]
            current_modal = mck_modal_properties(mass, stiffness, damping, valid, modes)
            mode_valid = current_modal["valid"]
            frequency_delta = (current_modal["log_frequency"] - observed_modal["log_frequency"]) * mode_valid
            damping_delta = torch.log(current_modal["damping"].clamp_min(1.0e-8)) - torch.log(observed_modal["damping"].clamp_min(1.0e-8))
            pred_shape = current_modal["shape"]
            obs_shape = observed_modal["shape"]
            shape_mask = valid[:, :, None] * mode_valid[:, None, :]
            pred_shape = pred_shape / (pred_shape.square().mul(shape_mask).sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8))
            obs_shape = obs_shape / (obs_shape.square().mul(shape_mask).sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8))
            sign = torch.sign((pred_shape * obs_shape * shape_mask).sum(dim=1, keepdim=True)).clamp(min=-1.0, max=1.0)
            sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
            shape_delta = (pred_shape - sign * obs_shape).abs() * shape_mask
            denom = mode_valid.sum(dim=1).clamp_min(1.0)
            modal_global = torch.stack([
                frequency_delta.sum(dim=1) / denom,
                shape_delta.sum(dim=(1, 2)) / (valid.sum(dim=1).clamp_min(1.0) * denom),
                (damping_delta * mode_valid).sum(dim=1) / denom,
            ], dim=-1)
            node_modal = shape_delta.sum(dim=2) / denom[:, None]
        mass_features = torch.cat([mass_latent, node_residual[:, :, None], node_modal[:, :, None], modal_global[:, None, :].expand(-1, mass.shape[1], -1)], dim=-1)
        mass_latent = self.mass_norm(mass_latent + self.mass_update(mass_features))
        mass_latent = mass_latent * valid[:, :, None]
        lower = torch.cat([torch.zeros_like(node_residual[:, :1]), node_residual[:, :-1]], dim=1)
        lower_modal = torch.cat([torch.zeros_like(node_modal[:, :1]), node_modal[:, :-1]], dim=1)
        edge_feature = torch.cat([
            edge_latent, node_residual[:, :, None], lower[:, :, None], node_modal[:, :, None],
            lower_modal[:, :, None], modal_global[:, None, :].expand(-1, mass.shape[1], -1),
        ], dim=-1)
        edge_latent = self.edge_norm(edge_latent + self.edge_update(edge_feature))
        edge_latent = edge_latent * structural_edge_mask(valid)[:, :, None].to(edge_latent)
        return floor, excitation, mass_latent, edge_latent


class PhysicsInformedHeteroGraphOperator(nn.Module):
    """Joint response-input-M/C/K inverse operator on a typed structural graph."""

    relation_names = (
        "structural_coupling",
        "global_operator",
        "sensor_to_excitation",
        "excitation_to_floor",
    )

    def __init__(self, config: HeteroGNOConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeteroGNOConfig()
        self.config.validate()
        hidden = self.config.hidden_dim
        self.floor_encoder = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.mass_encoder = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.excitation_token = nn.Parameter(torch.zeros(hidden))
        self.excitation_init = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.observation_set_conditioner = (
            ObservationSetConditioner(hidden, self.config.observation_distance_tau)
            if self.config.observation_set_conditioning else None
        )
        if self.config.hard_anchored_observation:
            self.observed_node_embedding = nn.Parameter(torch.zeros(hidden))
            self.missing_node_embedding = nn.Parameter(torch.zeros(hidden))
            self.missing_measurement_token = nn.Parameter(torch.zeros(hidden))
        else:
            self.register_parameter("observed_node_embedding", None)
            self.register_parameter("missing_node_embedding", None)
            self.register_parameter("missing_measurement_token", None)
        self.layout_attention_conditioner = (
            LayoutAttentionConditioner(self.config.depth, self.config.layout_condition_hidden_dim)
            if self.config.layout_conditioned_attention else None
        )
        self.sensor_density_conditioner = (
            SensorDensityConditioner(self.config.depth, self.config.sensor_density_hidden_dim)
            if self.config.sensor_density_conditioning else None
        )
        self.layout_feature_film = (
            LayoutFeatureFiLM(hidden, self.config.layout_condition_hidden_dim)
            if self.config.layout_feature_film else None
        )
        if self.config.measurement_type_conditioning:
            self.measurement_embedding = nn.Embedding(2, hidden)
            self.measurement_stem_a = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
            self.measurement_stem_v = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
            nn.init.zeros_(self.measurement_embedding.weight)
            nn.init.zeros_(self.measurement_stem_a[-1].weight)
            nn.init.zeros_(self.measurement_stem_a[-1].bias)
            nn.init.zeros_(self.measurement_stem_v[-1].weight)
            nn.init.zeros_(self.measurement_stem_v[-1].bias)
        else:
            self.measurement_embedding = None
            self.measurement_stem_a = None
            self.measurement_stem_v = None
        if self.config.time_aware_conditioning:
            self.physical_time_projection = nn.Linear(2 * self.config.time_fourier_bands, hidden)
            self.sampling_interval_projection = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
            nn.init.zeros_(self.physical_time_projection.weight)
            nn.init.zeros_(self.physical_time_projection.bias)
            nn.init.zeros_(self.sampling_interval_projection[-1].weight)
            nn.init.zeros_(self.sampling_interval_projection[-1].bias)
        else:
            self.physical_time_projection = None
            self.sampling_interval_projection = None
        if self.config.continuous_time_lifting:
            self.physical_lifting_projection = nn.Linear(hidden, hidden)
            nn.init.zeros_(self.physical_lifting_projection.weight)
            nn.init.zeros_(self.physical_lifting_projection.bias)
        else:
            self.physical_lifting_projection = None
        self.blocks = nn.ModuleList([HeterogeneousOperatorBlock(self.config) for _ in range(self.config.depth)])
        self.floor_history = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.floor_history_projection = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU())
        self.excitation_history = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.excitation_history_projection = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU())
        self.record_parameter_context = RecordParameterContext(hidden)
        self.mass_context_fusion = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU())
        self.edge_context_fusion = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU())
        self.observed_modal_encoder = ObservedModalEvidenceEncoder(hidden, num_modes=3)
        self.modal_floor_condition = nn.Linear(hidden, hidden)
        self.modal_mass_condition = nn.Linear(hidden, hidden)
        self.modal_edge_condition = nn.Linear(hidden, hidden)
        self.modal_excitation_condition = nn.Linear(hidden, hidden)
        self.modal_feature_encoder = (
            nn.Sequential(nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            if self.config.modal_feature_conditioning else None
        )
        self.physics_refinements = nn.ModuleList(
            [PhysicsGraphCorrection(hidden) for _ in range(self.config.physics_refinement_steps)]
        )
        self.response_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.input_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.full_field_input_bridge = (
            nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU())
            if self.config.full_field_conditioned_input else None
        )
        # Preserve a short path from the measured absolute accelerations to the
        # recovered excitation.  The graph blocks and bidirectional GRU are
        # effective for the structural response, but can smooth the common
        # high-frequency component shared by all observed floors.  Zero
        # initialization makes this branch an exact no-op for old checkpoints;
        # fine-tuning can then learn only the missing local correction.
        input_residual_hidden = max(hidden // 4, 16)
        self.observed_input_residual = nn.Sequential(
            nn.Conv1d(4, input_residual_hidden, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(input_residual_hidden, 1, kernel_size=9, padding=4),
        )
        nn.init.zeros_(self.observed_input_residual[-1].weight)
        nn.init.zeros_(self.observed_input_residual[-1].bias)
        self.parameter_decoder = BoundedMCKDecoder(hidden, self.config)

    def forward(
        self,
        sparse_acceleration: torch.Tensor,
        sensor_mask: torch.Tensor,
        coords: torch.Tensor,
        valid_node_mask: torch.Tensor,
        nominal_mass: torch.Tensor,
        nominal_stiffness: torch.Tensor,
        nominal_damping: torch.Tensor,
        total_mass_prior: torch.Tensor | None = None,
        parameter_context: torch.Tensor | None = None,
        modal_context: torch.Tensor | None = None,
        physics_dt: float | None = None,
        physics_normalizer: dict[str, object] | None = None,
        return_attention_audit: bool = False,
        measurement_type: torch.Tensor | None = None,
        return_latent_audit: bool = False,
        physical_dt: torch.Tensor | float | None = None,
        return_time_audit: bool = False,
        modal_relation_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if sparse_acceleration.ndim != 3:
            raise ValueError("sparse_acceleration must have shape [batch,time,nodes]")
        batch, steps, nodes = sparse_acceleration.shape
        valid = valid_node_mask.to(sparse_acceleration)
        sensor = sensor_mask.to(sparse_acceleration) * valid
        coords = _expand_coordinates(coords, batch, nodes, sparse_acceleration)
        nominal_mass = nominal_mass.to(sparse_acceleration)
        nominal_stiffness = nominal_stiffness.to(sparse_acceleration)
        nominal_damping = nominal_damping.to(sparse_acceleration)
        if nominal_mass.ndim == 1:
            nominal_mass = nominal_mass[None].expand(batch, -1)
            nominal_stiffness = nominal_stiffness[None].expand(batch, -1)
            nominal_damping = nominal_damping[None].expand(batch, -1)
        edge_valid = structural_edge_mask(valid).to(sparse_acceleration)
        modal_graph = None
        layout_descriptor = None
        density_descriptor = None
        layout_attention_delta = None
        if self.config.modal_guided_attention:
            if modal_relation_prior is None:
                modal_graph = modal_relation_graph(
                    nominal_mass, nominal_stiffness, nominal_damping, valid,
                    self.config.modal_graph_modes,
                )
            else:
                modal_graph = modal_relation_prior.to(sparse_acceleration) * valid[:, :, None] * valid[:, None, :]
            if self.layout_attention_conditioner is not None:
                layout_descriptor = layout_observability_descriptor(
                    sensor, coords, nominal_mass, nominal_stiffness, nominal_damping,
                    valid, self.config.modal_graph_modes,
                )
                layout_attention_delta = self.layout_attention_conditioner(layout_descriptor)
            if self.sensor_density_conditioner is not None:
                density_descriptor = sensor_density_descriptor(
                    sensor, coords, nominal_mass, nominal_stiffness, nominal_damping,
                    valid, self.config.modal_graph_modes,
                )
                density_delta = self.sensor_density_conditioner(density_descriptor)
                layout_attention_delta = (
                    density_delta if layout_attention_delta is None else layout_attention_delta + density_delta
                )
        time = torch.linspace(0.0, 1.0, steps, dtype=sparse_acceleration.dtype, device=sparse_acceleration.device)
        mass_log = _masked_standardize_log(nominal_mass, valid)
        stiffness_log = _masked_standardize_log(nominal_stiffness, edge_valid)
        damping_log = _masked_standardize_log(nominal_damping, edge_valid)

        structure_measurement = (
            torch.zeros_like(sparse_acceleration)
            if self.config.observation_set_conditioning
            else sparse_acceleration * sensor[:, None, :]
        )
        structure_sensor = torch.zeros_like(sensor) if self.config.observation_set_conditioning else sensor
        floor_features = torch.stack(
            [
                structure_measurement,
                structure_sensor[:, None, :].expand(-1, steps, -1),
                valid[:, None, :].expand(-1, steps, -1),
                coords[:, None, :].expand(-1, steps, -1),
                time[None, :, None].expand(batch, -1, nodes),
                mass_log[:, None, :].expand(-1, steps, -1),
            ],
            dim=-1,
        )
        floor = self.floor_encoder(floor_features) * valid[:, None, :, None]
        if self.config.hard_anchored_observation:
            observed = sensor[:, None, :, None]
            missing = (valid - sensor)[:, None, :, None]
            missing_features = floor_features.clone()
            missing_features[..., 0] = 0.0
            missing_features[..., 1] = 0.0
            missing_base = self.floor_encoder(missing_features)
            floor = (
                observed * (floor + self.observed_node_embedding.view(1, 1, 1, -1))
                + missing * (
                    missing_base
                    + self.missing_measurement_token.view(1, 1, 1, -1)
                    + self.missing_node_embedding.view(1, 1, 1, -1)
                )
            ) * valid[:, None, :, None]
        if self.config.measurement_type_conditioning:
            if measurement_type is None:
                measurement_type = torch.zeros((batch, nodes), dtype=torch.long, device=floor.device)
            else:
                measurement_type = measurement_type.to(device=floor.device, dtype=torch.long)
                if measurement_type.ndim == 1:
                    measurement_type = measurement_type[:, None].expand(-1, nodes)
                if measurement_type.shape != (batch, nodes):
                    raise ValueError("measurement_type must have shape [batch] or [batch,nodes]")
            measurement_value = sparse_acceleration[..., None]
            stem_a = self.measurement_stem_a(measurement_value)
            stem_v = self.measurement_stem_v(measurement_value)
            is_velocity = measurement_type[:, None, :, None].gt(0)
            type_correction = torch.where(is_velocity, stem_v, stem_a)
            type_embedding = self.measurement_embedding(measurement_type)[:, None, :, :]
            floor = floor + sensor[:, None, :, None] * (type_correction + type_embedding)
            floor = floor * valid[:, None, :, None]
        time_conditioning = None
        physical_time_features = None
        log_dt_ratio = None
        if self.config.time_aware_conditioning:
            if physical_dt is None:
                raise ValueError("time-aware conditioning requires physical_dt")
            dt_tensor = torch.as_tensor(physical_dt, dtype=floor.dtype, device=floor.device)
            if dt_tensor.ndim == 0:
                dt_tensor = dt_tensor.expand(batch)
            else:
                dt_tensor = dt_tensor.reshape(-1)
            if dt_tensor.shape != (batch,):
                raise ValueError("physical_dt must be scalar or have shape [batch]")
            physical_time = torch.arange(steps, dtype=floor.dtype, device=floor.device)[None, :] * dt_tensor[:, None]
            base_frequency = 1.0 / float(self.config.reference_window_duration)
            bands = torch.pow(
                floor.new_tensor(2.0), torch.arange(self.config.time_fourier_bands, device=floor.device, dtype=floor.dtype)
            ) * base_frequency
            phase = 2.0 * math.pi * physical_time[:, :, None] * bands[None, None, :]
            physical_time_features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
            log_dt_ratio = torch.log(dt_tensor / float(self.config.reference_dt)).clamp(-4.0, 4.0)
            time_conditioning = (
                self.physical_time_projection(physical_time_features)
                + self.sampling_interval_projection(log_dt_ratio[:, None])[:, None, :]
            )
            floor = (floor + time_conditioning[:, :, None, :]) * valid[:, None, :, None]
            if self.physical_lifting_projection is not None:
                # Rate-dependent discrete stencil for one fixed physical-time
                # neighborhood.  Batches are grouped by dt, so every sample in
                # the batch uses the same compact kernel without padding masks.
                if not torch.allclose(dt_tensor, dt_tensor[:1]):
                    raise ValueError("continuous-time lifting requires rate-grouped batches")
                dt_value = float(dt_tensor[0].detach().cpu())
                radius_samples = max(1, int(math.ceil(self.config.physical_lifting_radius / dt_value)))
                offsets = torch.arange(-radius_samples, radius_samples + 1, device=floor.device, dtype=floor.dtype)
                physical_offsets = offsets * dt_tensor[0]
                inside = physical_offsets.abs() <= float(self.config.physical_lifting_radius) + 1.0e-8
                sigma = float(self.config.physical_lifting_radius) / 2.0
                kernel = torch.exp(-0.5 * (physical_offsets / sigma).square()) * inside.to(floor)
                kernel = kernel / kernel.sum().clamp_min(1.0e-12)
                channels = floor.shape[-1]
                sequence = floor.permute(0, 2, 3, 1).reshape(batch * nodes, channels, steps)
                depthwise = kernel.view(1, 1, -1).expand(channels, 1, -1)
                smoothed = F.conv1d(sequence, depthwise, padding=radius_samples, groups=channels)
                denominator = F.conv1d(
                    torch.ones((batch * nodes, 1, steps), device=floor.device, dtype=floor.dtype),
                    kernel.view(1, 1, -1), padding=radius_samples,
                ).clamp_min(1.0e-6)
                smoothed = smoothed / denominator
                smoothed = smoothed.reshape(batch, nodes, channels, steps).permute(0, 3, 1, 2)
                floor = (floor + self.physical_lifting_projection(smoothed - floor)) * valid[:, None, :, None]
        if self.layout_feature_film is not None:
            if layout_descriptor is None:
                raise ValueError("layout FiLM requires a layout descriptor")
            floor = self.layout_feature_film(floor, layout_descriptor) * valid[:, None, :, None]
        observation_floor = None
        if self.config.observation_set_conditioning:
            observation_features = torch.stack(
                [
                    sparse_acceleration,
                    sensor[:, None, :].expand(-1, steps, -1),
                    valid[:, None, :].expand(-1, steps, -1),
                    coords[:, None, :].expand(-1, steps, -1),
                    time[None, :, None].expand(batch, -1, nodes),
                    mass_log[:, None, :].expand(-1, steps, -1),
                ],
                dim=-1,
            )
            observation_floor = self.floor_encoder(observation_features) * valid[:, None, :, None]
        mass_latent = self.mass_encoder(torch.stack([coords, valid, structure_sensor, mass_log], dim=-1)) * valid[:, :, None]
        edge_latent = self.edge_encoder(
            torch.stack([coords, edge_valid, stiffness_log, damping_log, 0.5 * (structure_sensor + torch.cat([structure_sensor[:, :1] * 0.0, structure_sensor[:, :-1]], dim=1))], dim=-1)
        ) * edge_valid[:, :, None]
        # E2-A1 control: nominal modal signatures are available only as
        # ordinary node/global features. They never alter attention scores or
        # message-passing relations (modal_guided_attention remains false).
        if self.modal_feature_encoder is not None:
            nominal_modes = mck_modal_properties(
                nominal_mass, nominal_stiffness, nominal_damping, valid,
                self.config.modal_graph_modes,
            )
            mode_count = self.config.modal_graph_modes
            log_frequency = nominal_modes["log_frequency"][:, :mode_count]
            damping = torch.log(nominal_modes["damping"][:, :mode_count].clamp_min(1.0e-8))
            shape = nominal_modes["shape"][:, :, :mode_count]
            signature = torch.cat([
                shape,
                log_frequency[:, None, :].expand(-1, nodes, -1),
                damping[:, None, :].expand(-1, nodes, -1),
            ], dim=-1)
            modal_feature = self.modal_feature_encoder(signature) * valid[:, :, None]
            floor = (floor + modal_feature[:, None, :, :]) * valid[:, None, :, None]
            mass_latent = (mass_latent + modal_feature) * valid[:, :, None]
            lower_modal = torch.cat([modal_feature[:, :1] * 0.0, modal_feature[:, :-1]], dim=1)
            edge_latent = (edge_latent + 0.5 * (modal_feature + lower_modal)) * edge_valid[:, :, None]
        if parameter_context is not None:
            parameter_context = parameter_context.to(sparse_acceleration)
            context_mass, context_edge = self.record_parameter_context(
                parameter_context, sensor, valid, coords, mass_log
            )
            mass_latent = self.mass_context_fusion(torch.cat([mass_latent, context_mass], dim=-1)) * valid[:, :, None]
            edge_latent = self.edge_context_fusion(torch.cat([edge_latent, context_edge], dim=-1)) * edge_valid[:, :, None]

        modal_outputs: dict[str, torch.Tensor] = {}
        modal_token = None
        if modal_context is not None:
            modal_context = modal_context.to(sparse_acceleration)
            modal_token, modal_log_frequency, modal_damping, modal_shape = self.observed_modal_encoder(
                modal_context, sensor, valid, coords
            )
            if self.config.modal_conditioning_scope == "all":
                floor = floor + self.modal_floor_condition(modal_token)[:, None, None, :]
                floor = floor * valid[:, None, :, None]
            mass_latent = (mass_latent + self.modal_mass_condition(modal_token)[:, None, :]) * valid[:, :, None]
            edge_latent = (edge_latent + self.modal_edge_condition(modal_token)[:, None, :]) * edge_valid[:, :, None]
            modal_outputs = {
                "observed_modal_log_frequency": modal_log_frequency,
                "observed_modal_damping": modal_damping,
                "observed_modal_shape": modal_shape,
            }

        observed_weight = sensor / sensor.sum(dim=1, keepdim=True).clamp_min(1.0)
        observation_set_weights = None
        if self.observation_set_conditioner is not None:
            if modal_graph is None:
                raise ValueError("observation-set conditioning requires modal-guided graph")
            floor, observed_pool, observation_set_weights = self.observation_set_conditioner(
                floor, sparse_acceleration, sensor, coords, modal_graph, valid, observation_floor
            )
        else:
            observed_pool = (floor * observed_weight[:, None, :, None]).sum(dim=2)
        observed_mean = (sparse_acceleration * observed_weight[:, None, :]).sum(dim=2)
        observed_centered = (sparse_acceleration - observed_mean[:, :, None]) * sensor[:, None, :]
        observed_variance = (
            observed_centered.square() * observed_weight[:, None, :]
        ).sum(dim=2)
        observed_rms = (
            sparse_acceleration.square() * observed_weight[:, None, :]
        ).sum(dim=2).clamp_min(1.0e-12).sqrt()
        observed_abs_mean = (
            sparse_acceleration.abs() * observed_weight[:, None, :]
        ).sum(dim=2)
        observed_input_features = torch.stack(
            [observed_mean, observed_variance.clamp_min(1.0e-12).sqrt(), observed_rms, observed_abs_mean],
            dim=1,
        )
        observed_input_residual = self.observed_input_residual(observed_input_features).squeeze(1)
        token = self.excitation_token.view(1, 1, -1).expand(batch, steps, -1)
        excitation = self.excitation_init(torch.cat([token, observed_pool], dim=-1))
        if modal_token is not None and self.config.modal_conditioning_scope == "all":
            excitation = excitation + self.modal_excitation_condition(modal_token)[:, None, :]
        attention_audits = []
        effective_biases = []
        layer_latent_audits = []
        if return_latent_audit:
            denominator = (valid.sum(dim=1) * steps).clamp_min(1.0)[:, None]
            layer_latent_audits.append((floor * valid[:, None, :, None]).sum(dim=(1, 2)) / denominator)
        for layer_index, block in enumerate(self.blocks):
            layer_delta = (
                layout_attention_delta[:, layer_index] if layout_attention_delta is not None else None
            )
            block_output = block(
                floor, excitation, mass_latent, edge_latent, coords, sensor, valid,
                modal_graph, layer_delta, return_attention_audit,
            )
            if layer_delta is not None:
                effective_biases.append(torch.stack([
                    block.global_relation.beta + layer_delta[:, 0],
                    block.global_relation.gamma + layer_delta[:, 1],
                ], dim=-1))
            if return_attention_audit:
                floor, excitation, mass_latent, edge_latent, attention_weights = block_output
                attention_audits.append(attention_weights)
            else:
                floor, excitation, mass_latent, edge_latent = block_output
            if return_latent_audit:
                layer_latent_audits.append((floor * valid[:, None, :, None]).sum(dim=(1, 2)) / denominator)

        floor_sequence = floor.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        floor_history, _ = self.floor_history(floor_sequence)
        floor_history = self.floor_history_projection(floor_history)
        floor_history = floor_history.reshape(batch, nodes, steps, -1).permute(0, 2, 1, 3)
        floor = (floor + floor_history) * valid[:, None, :, None]
        excitation_history, _ = self.excitation_history(excitation)
        excitation = excitation + self.excitation_history_projection(excitation_history)

        response = self.response_head(floor) * valid[:, None, :, None]
        direct_input_acceleration = self.input_head(excitation).squeeze(-1) + observed_input_residual
        full_field_input_latent = None
        if self.full_field_input_bridge is not None:
            node_weight = valid[:, None, :, None]
            node_count = node_weight.sum(dim=2).clamp_min(1.0)
            full_mean = (floor * node_weight).sum(dim=2) / node_count
            full_variance = ((floor - full_mean[:, :, None]).square() * node_weight).sum(dim=2) / node_count
            canonical_full_field = torch.cat([full_mean, full_variance.clamp_min(1.0e-12).sqrt()], dim=-1)
            if self.config.full_field_stop_gradient:
                canonical_full_field = canonical_full_field.detach()
            full_field_input_latent = self.full_field_input_bridge(canonical_full_field)
            input_acceleration = self.input_head(full_field_input_latent).squeeze(-1)
        else:
            input_acceleration = direct_input_acceleration
        parameters = self.parameter_decoder(
            mass_latent,
            edge_latent,
            nominal_mass,
            nominal_stiffness,
            nominal_damping,
            valid,
            total_mass_prior,
        )
        if self.physics_refinements:
            if physics_dt is None or physics_normalizer is None:
                raise ValueError("physics refinement requires physics_dt and physics_normalizer")
            for correction in self.physics_refinements:
                floor, excitation, mass_latent, edge_latent = correction(
                    floor, excitation, mass_latent, edge_latent, response, input_acceleration,
                    parameters["mass"], parameters["stiffness"], parameters["damping"], valid,
                    physics_dt, physics_normalizer,
                    {"log_frequency": modal_outputs["observed_modal_log_frequency"],
                     "damping": modal_outputs["observed_modal_damping"],
                     "shape": modal_outputs["observed_modal_shape"]} if modal_outputs else None,
                )
                response = self.response_head(floor) * valid[:, None, :, None]
                direct_input_acceleration = self.input_head(excitation).squeeze(-1) + observed_input_residual
                if self.full_field_input_bridge is not None:
                    node_weight = valid[:, None, :, None]
                    node_count = node_weight.sum(dim=2).clamp_min(1.0)
                    full_mean = (floor * node_weight).sum(dim=2) / node_count
                    full_variance = ((floor - full_mean[:, :, None]).square() * node_weight).sum(dim=2) / node_count
                    canonical_full_field = torch.cat([full_mean, full_variance.clamp_min(1.0e-12).sqrt()], dim=-1)
                    if self.config.full_field_stop_gradient:
                        canonical_full_field = canonical_full_field.detach()
                    full_field_input_latent = self.full_field_input_bridge(canonical_full_field)
                    input_acceleration = self.input_head(full_field_input_latent).squeeze(-1)
                else:
                    input_acceleration = direct_input_acceleration
                parameters = self.parameter_decoder(
                    mass_latent, edge_latent, nominal_mass, nominal_stiffness, nominal_damping, valid, total_mass_prior
                )
        mass_matrix, damping_matrix, stiffness_matrix = assemble_shear_mck(
            parameters["mass"], parameters["stiffness"], parameters["damping"], valid
        )
        result = {
            "response": response,
            "input": input_acceleration,
            "direct_input": direct_input_acceleration,
            "floor_latent": floor,
            "excitation_latent": excitation,
            "mass_latent": mass_latent,
            "edge_latent": edge_latent,
            "mass_matrix": mass_matrix,
            "damping_matrix": damping_matrix,
            "stiffness_matrix": stiffness_matrix,
            **modal_outputs,
            **parameters,
        }
        if full_field_input_latent is not None:
            result["full_field_input_latent"] = full_field_input_latent
        if modal_graph is not None:
            result["modal_relation_graph"] = modal_graph
        if layout_descriptor is not None:
            result["layout_descriptor"] = layout_descriptor
            result["layout_attention_biases"] = torch.stack(effective_biases, dim=1)
        if density_descriptor is not None:
            result["sensor_density_descriptor"] = density_descriptor
        if attention_audits:
            result["modal_attention_weights"] = torch.stack(attention_audits, dim=0)
        if observation_set_weights is not None:
            result["observation_set_attention"] = observation_set_weights
        if layer_latent_audits:
            result["layer_latent_audit"] = torch.stack(layer_latent_audits, dim=1)
        if return_time_audit and time_conditioning is not None:
            result["time_feature_mean"] = physical_time_features.mean(dim=(1, 2))
            result["time_feature_std"] = physical_time_features.std(dim=(1, 2), unbiased=False)
            result["time_conditioning_norm"] = time_conditioning.norm(dim=-1).mean(dim=1)
            result["log_dt_ratio"] = log_dt_ratio
        return result
