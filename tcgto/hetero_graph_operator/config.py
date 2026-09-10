from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class MassAnchorConfig:
    """Optional prior that fixes the otherwise ambiguous absolute mass scale."""

    enabled: bool = True
    enforcement: str = "exact"  # exact | soft
    value: float | None = None
    weight: float = 1.0

    def validate(self) -> None:
        if self.enforcement not in {"exact", "soft"}:
            raise ValueError("mass-anchor enforcement must be 'exact' or 'soft'")
        if self.value is not None and self.value <= 0.0:
            raise ValueError("mass-anchor value must be positive")
        if self.weight < 0.0:
            raise ValueError("mass-anchor weight must be non-negative")


@dataclass
class HeteroGNOConfig:
    hidden_dim: int = 64
    depth: int = 3
    num_heads: int = 4
    dropout: float = 0.05
    temporal_kernel_size: int = 5
    mass_log_bound: float = math.log(1.35)
    stiffness_log_bound: float = math.log(1.60)
    damping_log_bound: float = math.log(1.60)
    use_global_operator_edges: bool = True
    modal_conditioning_scope: str = "all"  # all | parameters
    physics_refinement_steps: int = 0
    modal_guided_attention: bool = False
    modal_feature_conditioning: bool = False
    modal_graph_modes: int = 3
    modal_local_bias_init: float = 0.05
    modal_relation_bias_init: float = 0.05
    observation_set_conditioning: bool = False
    observation_distance_tau: float = 0.25
    hard_anchored_observation: bool = False
    layout_conditioned_attention: bool = False
    layout_condition_hidden_dim: int = 16
    layout_feature_film: bool = False
    sensor_density_conditioning: bool = False
    sensor_density_hidden_dim: int = 16
    measurement_type_conditioning: bool = False
    time_aware_conditioning: bool = False
    reference_dt: float = 1.0
    reference_window_duration: float = 1.0
    time_fourier_bands: int = 4
    continuous_time_lifting: bool = False
    physical_lifting_radius: float = 0.25
    full_field_conditioned_input: bool = False
    full_field_stop_gradient: bool = True
    mass_anchor: MassAnchorConfig = field(default_factory=MassAnchorConfig)

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.depth <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_dim, depth and num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.temporal_kernel_size < 1 or self.temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        if self.modal_conditioning_scope not in {"all", "parameters"}:
            raise ValueError("modal_conditioning_scope must be 'all' or 'parameters'")
        if self.physics_refinement_steps < 0:
            raise ValueError("physics_refinement_steps must be non-negative")
        if self.modal_graph_modes < 1:
            raise ValueError("modal_graph_modes must be positive")
        if self.observation_distance_tau <= 0.0:
            raise ValueError("observation_distance_tau must be positive")
        if self.layout_condition_hidden_dim < 1:
            raise ValueError("layout_condition_hidden_dim must be positive")
        if self.observation_set_conditioning and self.hard_anchored_observation:
            raise ValueError("observation-set and hard-anchored conditioning are mutually exclusive")
        if self.layout_conditioned_attention and not self.modal_guided_attention:
            raise ValueError("layout-conditioned attention requires modal-guided attention")
        if self.sensor_density_hidden_dim < 1:
            raise ValueError("sensor_density_hidden_dim must be positive")
        if self.sensor_density_conditioning and not self.modal_guided_attention:
            raise ValueError("sensor-density conditioning requires modal-guided attention")
        if self.reference_dt <= 0.0 or self.reference_window_duration <= 0.0:
            raise ValueError("reference dt and window duration must be positive")
        if self.time_fourier_bands < 1:
            raise ValueError("time_fourier_bands must be positive")
        if self.physical_lifting_radius <= 0.0:
            raise ValueError("physical lifting radius must be positive")
        if self.continuous_time_lifting and not self.time_aware_conditioning:
            raise ValueError("continuous-time lifting requires time-aware conditioning")
        self.mass_anchor.validate()

    def to_dict(self) -> dict:
        return asdict(self)
