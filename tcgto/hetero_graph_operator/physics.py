from __future__ import annotations

from typing import Mapping

import torch


def shear_incidence(batch: int, nodes: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return B such that K = B^T diag(k_story) B.

    Story 0 connects the ground to floor 0. Story i>0 connects floors i-1
    and i. The same convention is used for damping edges.
    """

    base = torch.zeros(nodes, nodes, dtype=dtype, device=device)
    if nodes:
        base[0, 0] = 1.0
    if nodes > 1:
        idx = torch.arange(1, nodes, device=device)
        base[idx, idx - 1] = -1.0
        base[idx, idx] = 1.0
    return base.unsqueeze(0).expand(batch, -1, -1)


def structural_edge_mask(valid_node_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_node_mask.to(dtype=torch.bool)
    edge_valid = valid.clone()
    if valid.shape[1] > 1:
        edge_valid[:, 1:] = valid[:, 1:] & valid[:, :-1]
    return edge_valid


def assemble_shear_mck(
    mass: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    valid_node_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble batched exact M, C and K matrices from node/edge fields."""

    if not (mass.ndim == stiffness.ndim == damping.ndim == 2):
        raise ValueError("mass, stiffness and damping must have shape [batch,nodes]")
    if not (mass.shape == stiffness.shape == damping.shape):
        raise ValueError("mass, stiffness and damping shapes must match")
    batch, nodes = mass.shape
    if valid_node_mask is None:
        valid_node_mask = torch.ones_like(mass)
    valid = valid_node_mask.to(device=mass.device, dtype=mass.dtype)
    edge_valid = structural_edge_mask(valid_node_mask).to(dtype=mass.dtype)
    incidence = shear_incidence(batch, nodes, dtype=mass.dtype, device=mass.device)
    mass_matrix = torch.diag_embed(mass * valid)
    stiffness_matrix = incidence.transpose(1, 2) @ torch.diag_embed(stiffness * edge_valid) @ incidence
    damping_matrix = incidence.transpose(1, 2) @ torch.diag_embed(damping * edge_valid) @ incidence
    return mass_matrix, damping_matrix, stiffness_matrix


def mck_modal_properties(
    mass: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    valid_node_mask: torch.Tensor,
    num_modes: int,
) -> dict[str, torch.Tensor]:
    """Differentiable generalized-eigen modal quantities for shear buildings."""

    batch, nodes = mass.shape
    modes = min(max(int(num_modes), 1), nodes)
    _, damping_matrix, stiffness_matrix = assemble_shear_mck(mass, stiffness, damping, valid_node_mask)
    counts = valid_node_mask.sum(dim=1).to(torch.long)
    # The common fixed-DOF case is fully batched.  Besides being much faster on
    # CUDA, this avoids thousands of tiny eigensolver launches during training.
    if bool((counts == counts[0]).all().detach().cpu()):
        count = int(counts[0].detach().item())
        local_modes = min(modes, count)
        mass_local = mass[:, :count].clamp_min(1.0e-12)
        inv_sqrt = mass_local.rsqrt()
        normalized_stiffness = (
            inv_sqrt[:, :, None]
            * stiffness_matrix[:, :count, :count]
            * inv_sqrt[:, None, :]
        )
        eigenvalue, eigenvector = torch.linalg.eigh(normalized_stiffness)
        omega = eigenvalue[:, :local_modes].clamp_min(1.0e-12).sqrt()
        shape_local = inv_sqrt[:, :, None] * eigenvector[:, :, :local_modes]
        ratio = torch.diagonal(
            shape_local.transpose(1, 2) @ damping_matrix[:, :count, :count] @ shape_local,
            dim1=1,
            dim2=2,
        ) / (2.0 * omega)
        if local_modes < modes:
            pad_modes = modes - local_modes
            omega = torch.cat([omega, mass.new_zeros(batch, pad_modes)], dim=1)
            ratio = torch.cat([ratio, mass.new_zeros(batch, pad_modes)], dim=1)
            shape_local = torch.cat([shape_local, mass.new_zeros(batch, count, pad_modes)], dim=2)
        if count < nodes:
            shape_local = torch.cat([shape_local, mass.new_zeros(batch, nodes - count, modes)], dim=1)
        return {
            "log_frequency": torch.log(omega.clamp_min(1.0e-12)),
            "damping": ratio,
            "shape": shape_local,
            "valid": mass.new_ones(batch, modes),
        }
    frequency_rows, damping_rows, shape_rows, valid_rows = [], [], [], []
    for index in range(batch):
        count = int(valid_node_mask[index].sum().detach().item())
        mass_i = mass[index, :count].clamp_min(1.0e-12)
        inv_sqrt = mass_i.rsqrt()
        normalized_stiffness = inv_sqrt[:, None] * stiffness_matrix[index, :count, :count] * inv_sqrt[None, :]
        eigenvalue, eigenvector = torch.linalg.eigh(normalized_stiffness)
        local_modes = min(modes, count)
        omega = eigenvalue[:local_modes].clamp_min(1.0e-12).sqrt()
        shape = inv_sqrt[:, None] * eigenvector[:, :local_modes]
        ratio = torch.diagonal(shape.transpose(0, 1) @ damping_matrix[index, :count, :count] @ shape) / (2.0 * omega)
        frequency_pad = mass.new_zeros(modes)
        damping_pad = mass.new_zeros(modes)
        shape_pad = mass.new_zeros(nodes, modes)
        mode_valid = mass.new_zeros(modes)
        frequency_pad[:local_modes] = omega
        damping_pad[:local_modes] = ratio
        shape_pad[:count, :local_modes] = shape
        mode_valid[:local_modes] = 1.0
        frequency_rows.append(frequency_pad)
        damping_rows.append(damping_pad)
        shape_rows.append(shape_pad)
        valid_rows.append(mode_valid)
    return {
        "log_frequency": torch.log(torch.stack(frequency_rows).clamp_min(1.0e-12)),
        "damping": torch.stack(damping_rows),
        "shape": torch.stack(shape_rows),
        "valid": torch.stack(valid_rows),
    }


def _normalizer_tensor(
    normalizer: Mapping[str, object], name: str, reference: torch.Tensor
) -> torch.Tensor:
    return torch.as_tensor(normalizer[name], dtype=reference.dtype, device=reference.device)


def denormalize_response(
    response: torch.Tensor, normalizer: Mapping[str, object]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = _normalizer_tensor(normalizer, "response_mean", response)
    std = _normalizer_tensor(normalizer, "response_std", response)
    physical = response * std.view(1, 1, 1, 3) + mean.view(1, 1, 1, 3)
    return physical[..., 0], physical[..., 1], physical[..., 2]


def denormalize_input(input_acceleration: torch.Tensor, normalizer: Mapping[str, object]) -> torch.Tensor:
    return input_acceleration * float(normalizer["input_std"]) + float(normalizer["input_mean"])


def mck_physics_losses(
    response: torch.Tensor,
    input_acceleration: torch.Tensor,
    mass: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    valid_node_mask: torch.Tensor,
    dt: float,
    normalizer: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Return differentiable dynamic and kinematic consistency losses.

    The dynamic residual uses acceleration obtained from the velocity derivative,
    so the recovered ground motion remains explicitly present in the equation.
    This avoids the cancellation that occurs when absolute acceleration is used
    directly on both sides of the relative-coordinate equilibrium equation.
    """

    if response.shape[1] < 3:
        zero = response.new_tensor(0.0)
        return {"dynamic": zero, "kinematic": zero}
    residual, velocity_from_displacement, absolute_from_kinematics, velocity_mid, absolute_acceleration = mck_dynamic_residual(
        response, input_acceleration, mass, stiffness, damping, valid_node_mask, dt, normalizer
    )
    if residual.numel() == 0:
        zero = response.new_tensor(0.0)
        return {"dynamic": zero, "kinematic": zero}
    displacement, velocity, _ = denormalize_response(response, normalizer)
    step = max(float(dt), 1.0e-8)
    acceleration_scale = max(
        float(normalizer["response_std"][2]), float(normalizer["input_std"]), 1.0e-8
    )
    force_scale = (mass[:, None, :] * acceleration_scale).clamp_min(1.0e-8)
    valid = valid_node_mask[:, None, :].to(response)
    denom = valid.expand_as(residual).sum().clamp_min(1.0)
    dynamic = ((residual / force_scale).square() * valid).sum() / denom

    velocity_scale = max(float(normalizer["response_std"][1]), 1.0e-8)
    abs_acc_scale = max(float(normalizer["response_std"][2]), 1.0e-8)
    kinematic_velocity = ((velocity_from_displacement - velocity_mid) / velocity_scale).square()
    kinematic_acceleration = (
        (absolute_from_kinematics - absolute_acceleration[:, 1:-1]) / abs_acc_scale
    ).square()
    kinematic = ((kinematic_velocity + kinematic_acceleration) * valid).sum() / denom
    return {"dynamic": dynamic, "kinematic": kinematic}


def mck_dynamic_residual(
    response: torch.Tensor,
    input_acceleration: torch.Tensor,
    mass: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    valid_node_mask: torch.Tensor,
    dt: float,
    normalizer: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physical residual plus kinematic quantities used by an unrolled graph layer."""

    if response.shape[1] < 3:
        empty = response.new_zeros(response.shape[0], 0, response.shape[2])
        return empty, empty, empty, empty, empty
    displacement, velocity, absolute_acceleration = denormalize_response(response, normalizer)
    ground = denormalize_input(input_acceleration, normalizer)
    step = max(float(dt), 1.0e-8)
    velocity_mid = velocity[:, 1:-1]
    displacement_mid = displacement[:, 1:-1]
    ground_mid = ground[:, 1:-1]
    relative_acceleration = (velocity[:, 2:] - velocity[:, :-2]) / (2.0 * step)
    velocity_from_displacement = (displacement[:, 2:] - displacement[:, :-2]) / (2.0 * step)
    absolute_from_kinematics = relative_acceleration + ground_mid[:, :, None]
    mass_matrix, damping_matrix, stiffness_matrix = assemble_shear_mck(mass, stiffness, damping, valid_node_mask)
    residual = (
        torch.einsum("bij,btj->bti", mass_matrix, relative_acceleration)
        + torch.einsum("bij,btj->bti", damping_matrix, velocity_mid)
        + torch.einsum("bij,btj->bti", stiffness_matrix, displacement_mid)
        + mass[:, None, :] * ground_mid[:, :, None]
    )
    return residual, velocity_from_displacement, absolute_from_kinematics, velocity_mid, absolute_acceleration
