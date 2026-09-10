"""Physics-informed heterogeneous graph neural operator.

The implementation is intentionally dependency-light: it uses PyTorch only and
keeps physical structural edges separate from learned global operator edges.
"""

from .config import HeteroGNOConfig, MassAnchorConfig
from .model import PhysicsInformedHeteroGraphOperator
from .multiresolution_input import (
    HIGH_FREQUENCY_CUTOFF_HZ,
    MultiResolutionInputOperator,
    split_frequency_bands,
)
from .physics import assemble_shear_mck, mck_physics_losses

__all__ = [
    "HeteroGNOConfig",
    "MassAnchorConfig",
    "PhysicsInformedHeteroGraphOperator",
    "MultiResolutionInputOperator",
    "HIGH_FREQUENCY_CUTOFF_HZ",
    "split_frequency_bands",
    "assemble_shear_mck",
    "mck_physics_losses",
]
