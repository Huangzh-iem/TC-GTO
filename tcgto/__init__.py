"""TC-GTO model components."""

from .hetero_graph_operator.config import HeteroGNOConfig, MassAnchorConfig


def model_config() -> HeteroGNOConfig:
    return HeteroGNOConfig(
        hidden_dim=128,
        depth=5,
        num_heads=4,
        dropout=0.05,
        temporal_kernel_size=5,
        use_global_operator_edges=True,
        modal_guided_attention=False,
        modal_feature_conditioning=False,
        observation_set_conditioning=False,
        hard_anchored_observation=False,
        layout_conditioned_attention=False,
        layout_feature_film=False,
        sensor_density_conditioning=False,
        measurement_type_conditioning=False,
        time_aware_conditioning=False,
        continuous_time_lifting=False,
        full_field_conditioned_input=False,
        physics_refinement_steps=0,
        mass_anchor=MassAnchorConfig(enabled=False, weight=0.0),
    )
