from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .physics import (
    denormalize_input,
    denormalize_response,
    mck_modal_properties,
    mck_physics_losses,
    structural_edge_mask,
)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values)
    if values.ndim == 4 and mask.ndim == 2:
        mask = mask[:, None, :, None]
    elif values.ndim == 3 and mask.ndim == 2:
        mask = mask[:, None, :]
    else:
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def modal_supervision_loss(
    log_frequency: torch.Tensor,
    damping: torch.Tensor,
    shape: torch.Tensor,
    target_frequency: torch.Tensor,
    target_damping: torch.Tensor,
    target_shape: torch.Tensor,
    modal_valid: torch.Tensor,
    valid_nodes: torch.Tensor,
    damping_weight: float = 1.0,
) -> torch.Tensor:
    """Frequency/log-damping loss plus sign-invariant mode-shape MAC loss."""

    mode_mask = modal_valid.to(log_frequency)
    frequency_loss = ((log_frequency - torch.log(target_frequency.clamp_min(1.0e-12))).square() * mode_mask).sum()
    damping_loss = ((torch.log(damping.clamp_min(1.0e-8)) - torch.log(target_damping.clamp_min(1.0e-8))).square() * mode_mask).sum()
    frequency_loss = frequency_loss / mode_mask.sum().clamp_min(1.0)
    damping_loss = damping_loss / mode_mask.sum().clamp_min(1.0)
    node_mask = valid_nodes[:, :, None].to(shape) * mode_mask[:, None, :]
    dot = (shape * target_shape * node_mask).sum(dim=1)
    norm = (shape.square() * node_mask).sum(dim=1) * (target_shape.square() * node_mask).sum(dim=1)
    mac_loss = (((1.0 - dot.square() / norm.clamp_min(1.0e-12)) * mode_mask).sum() / mode_mask.sum().clamp_min(1.0))
    return frequency_loss + float(damping_weight) * damping_loss + mac_loss


def joint_inverse_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    normalizer: Mapping[str, object],
    dt: float,
    *,
    response_weight: float = 1.0,
    hidden_response_weight: float = 0.0,
    story_response_weight: float = 0.0,
    input_weight: float = 4.0,
    input_gradient_weight: float = 0.0,
    input_curvature_weight: float = 0.0,
    response_gradient_weight: float = 0.0,
    parameter_weight: float = 2.0,
    observation_weight: float = 0.5,
    dynamic_weight: float = 0.02,
    kinematic_weight: float = 0.05,
    mass_anchor_weight: float = 1.0,
    observed_modal_weight: float = 0.0,
    mck_modal_weight: float = 0.0,
    modal_consistency_weight: float = 0.0,
    modal_damping_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = batch["valid_node_mask"]
    edge_valid = structural_edge_mask(valid).to(valid)
    response = masked_mean((output["response"] - batch["response"]).square(), valid)
    hidden_mask = (1.0 - batch["mask"]) * valid
    hidden_response = masked_mean(
        (output["response"] - batch["response"]).square(), hidden_mask
    )
    # Downstream structural inversion is driven by interstory deformation and
    # velocity, not only by accurate absolute nodal traces.  Keep this optional
    # so historical experiments retain their exact objective.
    predicted_story = torch.diff(
        output["response"][..., :2], dim=2,
        prepend=torch.zeros_like(output["response"][:, :, :1, :2]),
    )
    target_story = torch.diff(
        batch["response"][..., :2], dim=2,
        prepend=torch.zeros_like(batch["response"][:, :, :1, :2]),
    )
    story_response = masked_mean((predicted_story - target_story).square(), valid)
    input_loss = F.mse_loss(output["input"], batch["input"])
    input_gradient = F.mse_loss(
        output["input"][:, 1:] - output["input"][:, :-1],
        batch["input"][:, 1:] - batch["input"][:, :-1],
    )
    input_curvature = F.mse_loss(
        output["input"][:, 2:] - 2.0 * output["input"][:, 1:-1] + output["input"][:, :-2],
        batch["input"][:, 2:] - 2.0 * batch["input"][:, 1:-1] + batch["input"][:, :-2],
    )
    response_gradient = masked_mean(
        (
            (output["response"][:, 1:] - output["response"][:, :-1])
            - (batch["response"][:, 1:] - batch["response"][:, :-1])
        ).square(),
        valid,
    )
    if parameter_weight > 0.0:
        parameter_terms = []
        for name, mask in [("mass", valid), ("stiffness", edge_valid), ("damping", edge_valid)]:
            log_error = torch.log(output[name].clamp_min(1.0e-12) / batch[name].clamp_min(1.0e-12))
            parameter_terms.append((log_error.square() * mask).sum() / mask.sum().clamp_min(1.0))
        parameter = torch.stack(parameter_terms).mean()
    else:
        parameter = response.new_zeros(())

    acceleration_mean = float(normalizer["acceleration_mean"])
    acceleration_std = float(normalizer["acceleration_std"])
    predicted_abs = output["response"][..., 2] * float(normalizer["response_std"][2]) + float(normalizer["response_mean"][2])
    predicted_abs_norm = (predicted_abs - acceleration_mean) / acceleration_std
    predicted_velocity_norm = output["response"][..., 1]
    measurement_type = batch.get("measurement_type")
    if measurement_type is None:
        predicted_measurement = predicted_abs_norm
    else:
        is_velocity = measurement_type[:, None, :].to(torch.bool)
        predicted_measurement = torch.where(is_velocity, predicted_velocity_norm, predicted_abs_norm)
    observed_mask = batch["mask"][:, None, :]
    observation = ((predicted_measurement - batch["sparse"]).square() * observed_mask).sum()
    observation = observation / observed_mask.expand_as(predicted_measurement).sum().clamp_min(1.0)
    if dynamic_weight > 0.0:
        physics = mck_physics_losses(
            output["response"],
            output["input"],
            output["mass"],
            output["stiffness"],
            output["damping"],
            valid,
            dt,
            normalizer,
        )
    elif kinematic_weight > 0.0 and output["response"].shape[1] >= 3:
        # Do not construct the M/C/K equilibrium graph when its weight is zero.
        # The former implementation calculated it and multiplied it by zero,
        # making response-only training several times slower and unnecessarily
        # routing a zero-valued graph through the parameter decoder.
        response_tensor = output["response"]
        displacement, velocity, absolute_acceleration = denormalize_response(response_tensor, normalizer)
        ground = denormalize_input(output["input"], normalizer)
        step = max(float(dt), 1.0e-8)
        velocity_from_displacement = (displacement[:, 2:] - displacement[:, :-2]) / (2.0 * step)
        relative_acceleration = (velocity[:, 2:] - velocity[:, :-2]) / (2.0 * step)
        absolute_from_kinematics = relative_acceleration + ground[:, 1:-1, None]
        valid_mid = valid[:, None, :].to(response_tensor)
        denominator = valid_mid.expand_as(velocity_from_displacement).sum().clamp_min(1.0)
        velocity_scale = max(float(normalizer["response_std"][1]), 1.0e-8)
        acceleration_scale = max(float(normalizer["response_std"][2]), 1.0e-8)
        kinematic = (
            ((velocity_from_displacement - velocity[:, 1:-1]) / velocity_scale).square()
            + ((absolute_from_kinematics - absolute_acceleration[:, 1:-1]) / acceleration_scale).square()
        )
        physics = {
            "dynamic": response_tensor.new_zeros(()),
            "kinematic": (kinematic * valid_mid).sum() / denominator,
        }
    else:
        zero = response.new_zeros(())
        physics = {"dynamic": zero, "kinematic": zero}
    anchor = output["mass_anchor_loss"] if mass_anchor_weight > 0.0 else response.new_zeros(())
    zero = response.new_zeros(())
    observed_modal = mck_modal = modal_consistency = zero
    if observed_modal_weight > 0.0 or mck_modal_weight > 0.0 or modal_consistency_weight > 0.0:
        if "modal_frequency" not in batch:
            raise ValueError("modal weights require modal_context_length > 0 in the dataset")
        target_args = (
            batch["modal_frequency"], batch["modal_damping"], batch["modal_shape"], batch["modal_valid"], valid
        )
        if observed_modal_weight > 0.0 or modal_consistency_weight > 0.0:
            if "observed_modal_log_frequency" not in output:
                raise ValueError("modal weights require modal_context input to the model")
            observed_modal = modal_supervision_loss(
                output["observed_modal_log_frequency"], output["observed_modal_damping"], output["observed_modal_shape"], *target_args,
                damping_weight=modal_damping_weight,
            )
        if mck_modal_weight > 0.0 or modal_consistency_weight > 0.0:
            mck_modal_values = mck_modal_properties(
                output["mass"], output["stiffness"], output["damping"], valid, batch["modal_frequency"].shape[1]
            )
            mck_modal = modal_supervision_loss(
                mck_modal_values["log_frequency"], mck_modal_values["damping"], mck_modal_values["shape"], *target_args,
                damping_weight=modal_damping_weight,
            )
        if modal_consistency_weight > 0.0:
            modal_consistency = modal_supervision_loss(
                mck_modal_values["log_frequency"], mck_modal_values["damping"], mck_modal_values["shape"],
                output["observed_modal_log_frequency"].exp(), output["observed_modal_damping"],
                output["observed_modal_shape"], batch["modal_valid"], valid,
                damping_weight=modal_damping_weight,
            )
    total = (
        response_weight * response
        + hidden_response_weight * hidden_response
        + story_response_weight * story_response
        + input_weight * input_loss
        + input_gradient_weight * input_gradient
        + input_curvature_weight * input_curvature
        + response_gradient_weight * response_gradient
        + parameter_weight * parameter
        + observation_weight * observation
        + dynamic_weight * physics["dynamic"]
        + kinematic_weight * physics["kinematic"]
        + mass_anchor_weight * anchor
        + observed_modal_weight * observed_modal
        + mck_modal_weight * mck_modal
        + modal_consistency_weight * modal_consistency
    )
    parts = {
        "total": total,
        "response": response,
        "hidden_response": hidden_response,
        "story_response": story_response,
        "input": input_loss,
        "input_gradient": input_gradient,
        "input_curvature": input_curvature,
        "response_gradient": response_gradient,
        "parameter": parameter,
        "observation": observation,
        "dynamic": physics["dynamic"],
        "kinematic": physics["kinematic"],
        "mass_anchor": anchor,
        "observed_modal": observed_modal,
        "mck_modal": mck_modal,
        "modal_consistency": modal_consistency,
    }
    return total, parts
