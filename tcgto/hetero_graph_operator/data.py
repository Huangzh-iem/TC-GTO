from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from generate_artificial_newmark_dataset import (
    SPECTRAL_FAMILIES,
    chain_matrix,
    fourier_shaped_motion,
    newmark_response,
)


@dataclass
class MCKRecord:
    name: str
    split: str
    num_dof: int
    time: np.ndarray
    ground: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    absolute_acceleration: np.ndarray
    nominal_mass: np.ndarray
    nominal_stiffness: np.ndarray
    nominal_damping: np.ndarray
    mass: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    structure_id: str = ""
    event_id: str = ""
    modal_frequency: np.ndarray | None = None
    modal_damping_ratio: np.ndarray | None = None
    modal_shape: np.ndarray | None = None
    floor_height: np.ndarray | None = None
    structure_family: str = ""
    metadata: dict | None = None


@dataclass
class MCKNormalizer:
    response_mean: np.ndarray
    response_std: np.ndarray
    acceleration_mean: float
    acceleration_std: float
    input_mean: float
    input_std: float

    def to_dict(self) -> dict[str, object]:
        return {
            "response_mean": self.response_mean.tolist(),
            "response_std": self.response_std.tolist(),
            "acceleration_mean": self.acceleration_mean,
            "acceleration_std": self.acceleration_std,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
        }


def response_array(record: MCKRecord) -> np.ndarray:
    return np.stack(
        [record.displacement, record.velocity, record.absolute_acceleration], axis=-1
    ).astype(np.float32)


def load_prototype_record(path: str) -> MCKRecord:
    """Load one D0 multi-system NPZ into the existing inverse-data interface."""
    import json

    with np.load(path, allow_pickle=False) as values:
        metadata = json.loads(str(values["metadata_json"].item()))
        return MCKRecord(
            name=str(values["record_id"].item()), split=str(metadata.get("split", "prototype")),
            num_dof=int(values["story_count"].item()), time=values["time"].astype(np.float32),
            ground=values["ground_motion"].astype(np.float32), displacement=values["q"].astype(np.float32),
            velocity=values["v"].astype(np.float32), absolute_acceleration=values["a_abs"].astype(np.float32),
            nominal_mass=values["nominal_mass"].astype(np.float32),
            nominal_stiffness=values["nominal_stiffness"].astype(np.float32),
            nominal_damping=values["nominal_damping"].astype(np.float32),
            mass=values["mass"].astype(np.float32), stiffness=values["stiffness"].astype(np.float32),
            damping=values["damping"].astype(np.float32), structure_id=str(values["structure_id"].item()),
            event_id=str(values["earthquake_id"].item()), modal_frequency=values["omega"].astype(np.float32),
            modal_damping_ratio=values["zeta"].astype(np.float32), modal_shape=values["phi"].astype(np.float32),
            floor_height=values["floor_height"].astype(np.float32),
            structure_family=str(values["structure_family"].item()), metadata=metadata,
        )


def compute_normalizer(records: list[MCKRecord]) -> MCKNormalizer:
    response = np.concatenate([response_array(item).reshape(-1, 3) for item in records], axis=0)
    acceleration = np.concatenate([item.absolute_acceleration.reshape(-1) for item in records])
    ground = np.concatenate([item.ground.reshape(-1) for item in records])
    return MCKNormalizer(
        response_mean=response.mean(axis=0).astype(np.float32),
        response_std=np.maximum(response.std(axis=0), 1.0e-8).astype(np.float32),
        acceleration_mean=float(acceleration.mean()),
        acceleration_std=max(float(acceleration.std()), 1.0e-8),
        input_mean=float(ground.mean()),
        input_std=max(float(ground.std()), 1.0e-8),
    )


def modal_targets(
    mass: np.ndarray, stiffness: np.ndarray, damping: np.ndarray, num_modes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mass-normalized modal frequency, damping ratio and mode shapes.

    These are *training targets* derived from the known synthetic M/C/K only.
    They are never provided to the inference model.
    """

    nodes = len(mass)
    modes = min(max(int(num_modes), 1), nodes)
    inv_sqrt_mass = 1.0 / np.sqrt(np.maximum(mass, 1.0e-12))
    stiffness_matrix = chain_matrix(stiffness)
    damping_matrix = chain_matrix(damping)
    eigval, eigvec = np.linalg.eigh(
        inv_sqrt_mass[:, None] * stiffness_matrix * inv_sqrt_mass[None, :]
    )
    omega = np.sqrt(np.maximum(eigval[:modes], 1.0e-12))
    shape = inv_sqrt_mass[:, None] * eigvec[:, :modes]
    for mode in range(modes):
        pivot = int(np.argmax(np.abs(shape[:, mode])))
        if shape[pivot, mode] < 0.0:
            shape[:, mode] *= -1.0
    modal_damping = np.diag(shape.T @ damping_matrix @ shape) / (2.0 * omega)
    return omega.astype(np.float32), modal_damping.astype(np.float32), shape.astype(np.float32)


def nominal_story_parameters(num_dof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord = np.linspace(0.0, 1.0, num_dof)
    mass = 2.0e4 * (1.04 - 0.08 * coord)
    stiffness = 1.65e7 * (1.22 - 0.44 * coord) * (1.0 + 0.04 * np.sin(1.7 * np.pi * coord))
    damping = 2.0 * 0.035 * np.sqrt(stiffness * mass)
    return mass.astype(np.float64), stiffness.astype(np.float64), damping.astype(np.float64)


def _smooth_log_profile(rng: np.random.Generator, nodes: int, sigma: float, bound: float) -> np.ndarray:
    coord = np.linspace(-1.0, 1.0, nodes)
    values = rng.normal(0.0, 0.45 * sigma)
    values = values + rng.normal(0.0, 0.30 * sigma) * coord
    values = values + rng.normal(0.0, 0.40 * sigma) * np.sin(rng.uniform(0.6, 1.4) * np.pi * (coord + 1.0))
    values = values + rng.normal(0.0, 0.12 * sigma, size=nodes)
    return np.clip(values, -bound, bound)


def generate_mck_records(
    *,
    train_dofs: list[int],
    test_dof: int,
    train_structures_per_dof: int,
    val_structures_per_dof: int,
    test_structures: int,
    events_per_structure: int,
    duration: float,
    steps: int,
    seed: int,
) -> list[MCKRecord]:
    """Generate a linear multi-structure dataset with independently varying M/K/C."""

    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, duration, steps, dtype=np.float64)
    dt = float(time[1] - time[0])
    records: list[MCKRecord] = []
    split_specs = [
        ("train", train_dofs, train_structures_per_dof),
        ("val", train_dofs, val_structures_per_dof),
        ("test", [test_dof], test_structures),
    ]
    structure_index = 0
    for split, dof_values, structure_count in split_specs:
        for num_dof in dof_values:
            for _ in range(structure_count):
                mass0, stiffness0, damping0 = nominal_story_parameters(num_dof)
                mass = mass0 * np.exp(_smooth_log_profile(rng, num_dof, 0.16, np.log(1.35)))
                stiffness = stiffness0 * np.exp(_smooth_log_profile(rng, num_dof, 0.28, np.log(1.60)))
                damping = damping0 * np.exp(_smooth_log_profile(rng, num_dof, 0.32, np.log(1.60)))
                mass_matrix = np.diag(mass)
                stiffness_matrix = chain_matrix(stiffness)
                damping_matrix = chain_matrix(damping)
                for event_index in range(events_per_structure):
                    family = SPECTRAL_FAMILIES[(structure_index * events_per_structure + event_index) % len(SPECTRAL_FAMILIES)]
                    ground, _ = fourier_shaped_motion(time, family, rng)
                    displacement, velocity, _, absolute_acceleration = newmark_response(
                        ground, dt, mass_matrix, damping_matrix, stiffness_matrix
                    )
                    records.append(
                        MCKRecord(
                            name=f"{split}_n{num_dof}_s{structure_index:03d}_e{event_index:02d}_{family}",
                            split=split,
                            num_dof=num_dof,
                            time=time.astype(np.float32),
                            ground=ground.astype(np.float32),
                            displacement=displacement,
                            velocity=velocity,
                            absolute_acceleration=absolute_acceleration,
                            nominal_mass=mass0.astype(np.float32),
                            nominal_stiffness=stiffness0.astype(np.float32),
                            nominal_damping=damping0.astype(np.float32),
                            mass=mass.astype(np.float32),
                            stiffness=stiffness.astype(np.float32),
                            damping=damping.astype(np.float32),
                        )
                    )
                structure_index += 1
    return records


def generate_controlled_overfit_records(
    *,
    num_dof: int = 8,
    events: int = 3,
    duration: float = 3.0,
    steps: int = 120,
    seed: int = 20260804,
) -> list[MCKRecord]:
    """Create a deliberately easy fixed-structure capacity/implementation test.

    Train, validation and test contain the same fixed M/C/K and the same event
    histories. This is not a generalization benchmark; it answers only whether
    the architecture and losses can represent the desired inverse map.
    """

    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, duration, steps, dtype=np.float64)
    dt = float(time[1] - time[0])
    mass0, stiffness0, damping0 = nominal_story_parameters(num_dof)
    profile_coord = np.linspace(0.0, 1.0, num_dof)
    template_coord = np.linspace(0.0, 1.0, 8)
    mass_ratio = np.interp(profile_coord, template_coord, [1.14, 1.11, 1.08, 1.05, 0.95, 0.92, 0.89, 0.86])
    stiffness_ratio = np.interp(profile_coord, template_coord, [1.26, 1.20, 1.13, 1.06, 0.92, 0.82, 0.76, 0.86])
    damping_ratio = np.interp(profile_coord, template_coord, [0.72, 0.78, 0.87, 0.96, 1.10, 1.24, 1.34, 1.18])
    mass = mass0 * mass_ratio
    stiffness = stiffness0 * stiffness_ratio
    damping = damping0 * damping_ratio
    mass_matrix = np.diag(mass)
    stiffness_matrix = chain_matrix(stiffness)
    damping_matrix = chain_matrix(damping)
    event_data = []
    for event_index in range(events):
        family = SPECTRAL_FAMILIES[event_index % len(SPECTRAL_FAMILIES)]
        ground, _ = fourier_shaped_motion(time, family, rng)
        displacement, velocity, _, absolute_acceleration = newmark_response(
            ground, dt, mass_matrix, damping_matrix, stiffness_matrix
        )
        event_data.append((family, ground, displacement, velocity, absolute_acceleration))

    records = []
    for split in ("train", "val", "test"):
        for event_index, (family, ground, displacement, velocity, absolute_acceleration) in enumerate(event_data):
            records.append(
                MCKRecord(
                    name=f"controlled_{split}_n{num_dof}_e{event_index:02d}_{family}",
                    split=split,
                    num_dof=num_dof,
                    time=time.astype(np.float32),
                    ground=ground.astype(np.float32),
                    displacement=displacement.copy(),
                    velocity=velocity.copy(),
                    absolute_acceleration=absolute_acceleration.copy(),
                    nominal_mass=mass0.astype(np.float32),
                    nominal_stiffness=stiffness0.astype(np.float32),
                    nominal_damping=damping0.astype(np.float32),
                    mass=mass.astype(np.float32),
                    stiffness=stiffness.astype(np.float32),
                    damping=damping.astype(np.float32),
                )
            )
    return records


def _profile_at_target_error(
    rng: np.random.Generator,
    nodes: int,
    target_error: float,
    log_bound: float,
) -> np.ndarray:
    raw = _smooth_log_profile(rng, nodes, 1.0, log_bound)
    signs = np.where(raw >= 0.0, 1.0, -1.0)
    raw = signs * np.maximum(np.abs(raw), 0.025)
    for _ in range(6):
        ratio = np.exp(raw)
        current = float(np.mean(np.abs(1.0 - ratio) / ratio))
        raw = np.clip(raw * target_error / max(current, 1.0e-8), -0.95 * log_bound, 0.95 * log_bound)
    return np.exp(raw)


def generate_generalization_protocol_records(
    *,
    num_dof: int = 8,
    train_structures: int = 16,
    val_structures: int = 4,
    test_structures: int = 6,
    train_events: int = 4,
    val_events: int = 2,
    test_events: int = 3,
    duration: float = 3.0,
    steps: int = 96,
    seed: int = 20260804,
    event_family_mode: str = "matched",
    pairing_mode: str = "cartesian",
    train_structures_per_event: int = 1,
) -> tuple[list[MCKRecord], dict[str, object]]:
    """Build disjoint structure/event banks and A/B/C generalization splits.

    ``matched`` keeps the earthquake IDs and waveforms disjoint while exposing
    every split to the same spectral-family distribution. ``heldout`` assigns
    families sequentially across groups and is intended as a harder OOD test.
    """

    if event_family_mode not in {"matched", "heldout"}:
        raise ValueError("event_family_mode must be 'matched' or 'heldout'")
    if pairing_mode not in {"cartesian", "balanced"}:
        raise ValueError("pairing_mode must be 'cartesian' or 'balanced'")
    if train_structures_per_event < 1:
        raise ValueError("train_structures_per_event must be at least 1")

    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, duration, steps, dtype=np.float64)
    dt = float(time[1] - time[0])
    mass0, stiffness0, damping0 = nominal_story_parameters(num_dof)

    structure_specs = [
        ("train", train_structures),
        ("val", val_structures),
        ("test", test_structures),
    ]
    structures: dict[str, dict[str, object]] = {}
    structure_groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    structure_rows = []
    for group, count in structure_specs:
        for index in range(count):
            structure_id = f"S_{group}_{index:03d}"
            mass_target = float(rng.uniform(0.06, 0.14))
            stiffness_target = float(rng.uniform(0.10, 0.24))
            damping_target = float(rng.uniform(0.14, 0.28))
            mass = mass0 * _profile_at_target_error(rng, num_dof, mass_target, np.log(1.35))
            stiffness = stiffness0 * _profile_at_target_error(rng, num_dof, stiffness_target, np.log(1.60))
            damping = damping0 * _profile_at_target_error(rng, num_dof, damping_target, np.log(1.60))
            structures[structure_id] = {
                "mass": mass,
                "stiffness": stiffness,
                "damping": damping,
            }
            structure_groups[group].append(structure_id)
            structure_rows.append(
                {
                    "structure_id": structure_id,
                    "group": group,
                    "mass_initial_error": float(np.mean(np.abs(mass0 - mass) / mass)),
                    "stiffness_initial_error": float(np.mean(np.abs(stiffness0 - stiffness) / stiffness)),
                    "damping_initial_error": float(np.mean(np.abs(damping0 - damping) / damping)),
                    "total_mass": float(np.sum(mass)),
                }
            )

    event_specs = [("train", train_events), ("val", val_events), ("test", test_events)]
    events: dict[str, np.ndarray] = {}
    event_groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    event_rows = []
    family_cursor = 0
    for group, count in event_specs:
        for index in range(count):
            event_id = f"E_{group}_{index:03d}"
            family_index = index if event_family_mode == "matched" else family_cursor
            family = SPECTRAL_FAMILIES[family_index % len(SPECTRAL_FAMILIES)]
            family_cursor += 1
            ground, metadata = fourier_shaped_motion(time, family, rng)
            events[event_id] = ground.astype(np.float32)
            event_groups[group].append(event_id)
            event_rows.append(
                {
                    "event_id": event_id,
                    "group": group,
                    "family": family,
                    "pga": float(metadata["pga"]),
                    "predominant_frequency": float(metadata["predominant_frequency"]),
                }
            )

    if pairing_mode == "cartesian":
        split_pairs = {
            "train": [(s, e) for s in structure_groups["train"] for e in event_groups["train"]],
            "val": [(s, e) for s in structure_groups["val"] for e in event_groups["val"]],
            "test_A": [
                (s, e)
                for s in structure_groups["train"][:test_structures]
                for e in event_groups["test"]
            ],
            "test_B": [
                (s, e)
                for s in structure_groups["test"]
                for e in event_groups["train"][:test_events]
            ],
            "test_C": [(s, e) for s in structure_groups["test"] for e in event_groups["test"]],
        }
    else:
        def balanced_pairs(
            structure_ids: list[str],
            event_ids: list[str],
            offset: int = 0,
            structures_per_event: int = 1,
        ) -> list[tuple[str, str]]:
            repeats = min(int(structures_per_event), len(structure_ids))
            return [
                (structure_ids[(index + offset + repeat) % len(structure_ids)], event_id)
                for index, event_id in enumerate(event_ids)
                for repeat in range(repeats)
            ]

        split_pairs = {
            "train": balanced_pairs(
                structure_groups["train"],
                event_groups["train"],
                structures_per_event=train_structures_per_event,
            ),
            "val": balanced_pairs(structure_groups["val"], event_groups["val"]),
            "test_A": balanced_pairs(structure_groups["train"], event_groups["test"], offset=1),
            "test_B": balanced_pairs(
                structure_groups["test"], event_groups["train"][:test_events], offset=1
            ),
            "test_C": balanced_pairs(structure_groups["test"], event_groups["test"], offset=2),
        }
    records = []
    pair_rows = []
    for split, pairs in split_pairs.items():
        for structure_id, event_id in pairs:
            values = structures[structure_id]
            mass = np.asarray(values["mass"], dtype=np.float64)
            stiffness = np.asarray(values["stiffness"], dtype=np.float64)
            damping = np.asarray(values["damping"], dtype=np.float64)
            ground = events[event_id]
            displacement, velocity, _, absolute_acceleration = newmark_response(
                ground,
                dt,
                np.diag(mass),
                chain_matrix(damping),
                chain_matrix(stiffness),
            )
            records.append(
                MCKRecord(
                    name=f"{split}_{structure_id}_{event_id}",
                    split=split,
                    num_dof=num_dof,
                    time=time.astype(np.float32),
                    ground=ground.copy(),
                    displacement=displacement,
                    velocity=velocity,
                    absolute_acceleration=absolute_acceleration,
                    nominal_mass=mass0.astype(np.float32),
                    nominal_stiffness=stiffness0.astype(np.float32),
                    nominal_damping=damping0.astype(np.float32),
                    mass=mass.astype(np.float32),
                    stiffness=stiffness.astype(np.float32),
                    damping=damping.astype(np.float32),
                    structure_id=structure_id,
                    event_id=event_id,
                )
            )
            pair_rows.append({"split": split, "structure_id": structure_id, "event_id": event_id})

    train_structure_set = set(structure_groups["train"])
    val_structure_set = set(structure_groups["val"])
    test_structure_set = set(structure_groups["test"])
    train_event_set = set(event_groups["train"])
    val_event_set = set(event_groups["val"])
    test_event_set = set(event_groups["test"])
    leakage_audit = {
        "train_val_structure_overlap": sorted(train_structure_set & val_structure_set),
        "train_test_structure_overlap": sorted(train_structure_set & test_structure_set),
        "val_test_structure_overlap": sorted(val_structure_set & test_structure_set),
        "train_val_event_overlap": sorted(train_event_set & val_event_set),
        "train_test_event_overlap": sorted(train_event_set & test_event_set),
        "val_test_event_overlap": sorted(val_event_set & test_event_set),
    }
    leakage_audit["passed"] = not any(leakage_audit.values())
    if not leakage_audit["passed"]:
        raise RuntimeError(f"Generalization split leakage detected: {leakage_audit}")
    manifest = {
        "event_family_mode": event_family_mode,
        "pairing_mode": pairing_mode,
        "train_structures_per_event": int(train_structures_per_event),
        "structure_rows": structure_rows,
        "event_rows": event_rows,
        "pair_rows": pair_rows,
        "structure_groups": structure_groups,
        "event_groups": event_groups,
        "split_counts": {name: len(pairs) for name, pairs in split_pairs.items()},
        "leakage_audit": leakage_audit,
    }
    return records, manifest


class MCKInverseDataset(Dataset):
    def __init__(
        self,
        records: list[MCKRecord],
        normalizer: MCKNormalizer,
        max_dof: int,
        min_sensors: int = 2,
        max_sensors: int = 4,
        noise_level: float = 0.0,
        fixed_observed_indices: list[int] | None = None,
        preferred_observed_indices: list[int] | None = None,
        fixed_mask_probability: float = 0.0,
        resample_each_call: bool = True,
        seed: int = 0,
        window_length: int | None = None,
        window_stride: int | None = None,
        parameter_context_length: int = 0,
        modal_context_length: int = 0,
        num_modal_modes: int = 3,
        allowed_observed_layouts: list[tuple[int, ...]] | None = None,
        measurement_mode: str = "acceleration",
    ) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        self.records = records
        self.normalizer = normalizer
        self.max_dof = int(max_dof)
        self.min_sensors = int(min_sensors)
        self.max_sensors = int(max_sensors)
        self.noise_level = float(noise_level)
        self.fixed_observed_indices = None if fixed_observed_indices is None else sorted(set(int(i) for i in fixed_observed_indices))
        self.preferred_observed_indices = None if preferred_observed_indices is None else sorted(set(int(i) for i in preferred_observed_indices))
        self.fixed_mask_probability = float(fixed_mask_probability)
        self.resample_each_call = bool(resample_each_call)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.parameter_context_length = max(int(parameter_context_length), 0)
        self.modal_context_length = max(int(modal_context_length), 0)
        self.num_modal_modes = max(int(num_modal_modes), 1)
        self.allowed_observed_layouts = None if allowed_observed_layouts is None else [
            tuple(sorted(int(index) for index in layout)) for layout in allowed_observed_layouts
        ]
        self.measurement_mode = str(measurement_mode)
        valid_measurement_modes = {
            "acceleration", "velocity", "random_homogeneous", "mixed_training",
            "acceleration_dominant", "balanced_mixed", "velocity_dominant",
            "extreme_one_acceleration", "extreme_one_velocity",
        }
        if self.measurement_mode not in valid_measurement_modes:
            raise ValueError(f"unsupported measurement_mode: {self.measurement_mode}")
        if self.allowed_observed_layouts is not None:
            if not self.allowed_observed_layouts:
                raise ValueError("allowed_observed_layouts cannot be empty")
            if any(len(layout) != len(set(layout)) for layout in self.allowed_observed_layouts):
                raise ValueError("allowed sensor layouts must contain unique indices")
        self.window_index: list[tuple[int, int, int]] = []
        for record_index, record in enumerate(records):
            steps = len(record.time)
            length = steps if window_length is None or window_length <= 0 else min(int(window_length), steps)
            stride = length if window_stride is None or window_stride <= 0 else int(window_stride)
            starts = list(range(0, steps - length + 1, stride))
            final_start = steps - length
            if starts[-1] != final_start:
                starts.append(final_start)
            self.window_index.extend((record_index, start, start + length) for start in starts)

    def __len__(self) -> int:
        return len(self.window_index)

    def _padded(self, values: np.ndarray, nodes: int, *, fill: float = 0.0) -> np.ndarray:
        shape = list(values.shape)
        node_axis = 1 if values.ndim > 1 else 0
        shape[node_axis] = self.max_dof
        padded = np.full(shape, fill, dtype=np.float32)
        if values.ndim == 1:
            padded[:nodes] = values
        else:
            padded[:, :nodes] = values
        return padded

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, start, end = self.window_index[index]
        record = self.records[record_index]
        rng = self.rng if self.resample_each_call else np.random.default_rng(self.seed + 104729 * int(index))
        nodes = record.num_dof
        valid = np.zeros(self.max_dof, dtype=np.float32)
        valid[:nodes] = 1.0
        if self.allowed_observed_layouts is not None:
            valid_layouts = [layout for layout in self.allowed_observed_layouts if all(0 <= item < nodes for item in layout)]
            if not valid_layouts:
                raise ValueError("allowed_observed_layouts contains no layout valid for this record")
            selected = valid_layouts[int(rng.integers(0, len(valid_layouts)))]
            observed = np.asarray(selected, dtype=np.int64)
            count = int(observed.size)
        elif self.fixed_observed_indices is not None:
            observed = np.asarray([i for i in self.fixed_observed_indices if 0 <= i < nodes], dtype=np.int64)
            if observed.size == 0:
                raise ValueError("fixed_observed_indices contains no valid floor for this record")
            count = int(observed.size)
        else:
            preferred = [] if self.preferred_observed_indices is None else [
                i for i in self.preferred_observed_indices if 0 <= i < nodes
            ]
            if preferred and rng.random() < self.fixed_mask_probability:
                observed = np.asarray(preferred, dtype=np.int64)
                count = int(observed.size)
            else:
                count = min(max(self.min_sensors, 1), nodes)
                upper = min(max(self.max_sensors, count), nodes)
                count = int(rng.integers(count, upper + 1))
                observed = rng.choice(np.arange(nodes), size=count, replace=False)
        mask = np.zeros(self.max_dof, dtype=np.float32)
        mask[observed] = 1.0
        measurement_type = np.zeros(self.max_dof, dtype=np.int64)  # 0 absolute-a, 1 relative-v
        mode = self.measurement_mode
        if mode == "velocity":
            measurement_type[observed] = 1
        elif mode == "random_homogeneous":
            measurement_type[observed] = int(rng.integers(0, 2))
        elif mode == "mixed_training":
            category = int(rng.integers(0, 3))
            if category == 1:
                measurement_type[observed] = 1
            elif category == 2:
                measurement_type[observed] = rng.integers(0, 2, size=count)
                if count > 1 and len(set(measurement_type[observed].tolist())) == 1:
                    measurement_type[observed[0]] = 1 - measurement_type[observed[0]]
        elif mode in {"acceleration_dominant", "balanced_mixed", "velocity_dominant"}:
            probability_velocity = {"acceleration_dominant": 0.25, "balanced_mixed": 0.50, "velocity_dominant": 0.75}[mode]
            measurement_type[observed] = (rng.random(count) < probability_velocity).astype(np.int64)
            if count > 1 and len(set(measurement_type[observed].tolist())) == 1:
                measurement_type[observed[0]] = 1 - measurement_type[observed[0]]
        elif mode == "extreme_one_acceleration":
            measurement_type[observed] = 1
            measurement_type[observed[0]] = 0
        elif mode == "extreme_one_velocity":
            measurement_type[observed] = 0
            measurement_type[observed[0]] = 1

        full_measurement = np.zeros((len(record.time), self.max_dof), dtype=np.float32)
        for type_code, source, mean, std in (
            (0, record.absolute_acceleration, self.normalizer.acceleration_mean, self.normalizer.acceleration_std),
            (1, record.velocity, self.normalizer.response_mean[1], self.normalizer.response_std[1]),
        ):
            selected = observed[measurement_type[observed] == type_code]
            if selected.size == 0:
                continue
            values = source[:, selected].copy()
            if self.noise_level > 0.0:
                scale = np.maximum(values.std(axis=0, keepdims=True), 1.0e-8)
                values += rng.normal(0.0, self.noise_level, size=values.shape).astype(np.float32) * scale
            full_measurement[:, selected] = ((values - mean) / std).astype(np.float32)
        sparse = full_measurement[start:end] * mask[None, :]
        response = self._padded(response_array(record)[start:end], nodes)
        response = (response - self.normalizer.response_mean[None, None, :]) / self.normalizer.response_std[None, None, :]
        response *= valid[None, :, None]
        ground = (record.ground[start:end] - self.normalizer.input_mean) / self.normalizer.input_std
        coords = np.zeros(self.max_dof, dtype=np.float32)
        coords[:nodes] = np.linspace(0.0, 1.0, nodes, dtype=np.float32)

        sample = {
            "sparse": torch.from_numpy(sparse.astype(np.float32)),
            "response": torch.from_numpy(response.astype(np.float32)),
            "input": torch.from_numpy(ground.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "valid_node_mask": torch.from_numpy(valid),
            "coords": torch.from_numpy(coords),
            "nominal_mass": torch.from_numpy(self._padded(record.nominal_mass, nodes)),
            "nominal_stiffness": torch.from_numpy(self._padded(record.nominal_stiffness, nodes)),
            "nominal_damping": torch.from_numpy(self._padded(record.nominal_damping, nodes)),
            "mass": torch.from_numpy(self._padded(record.mass, nodes)),
            "stiffness": torch.from_numpy(self._padded(record.stiffness, nodes)),
            "damping": torch.from_numpy(self._padded(record.damping, nodes)),
            "total_mass": torch.tensor(float(record.mass.sum()), dtype=torch.float32),
            "num_dof": torch.tensor(nodes, dtype=torch.long),
            "record_id": torch.tensor(record_index, dtype=torch.long),
            "measurement_type": torch.from_numpy(measurement_type),
        }
        # Numerical multisystem datasets may carry authoritative OpenSees mode
        # shapes that cannot be recovered from the legacy shear-chain surrogate.
        # Expose their parameter-free floor relation while preserving old data.
        if record.modal_shape is not None:
            shape = np.asarray(record.modal_shape, dtype=np.float32)[:nodes, : self.num_modal_modes]
            signature = shape / np.maximum(np.linalg.norm(shape, axis=0, keepdims=True), 1.0e-12)
            signature = signature / np.sqrt(float(max(signature.shape[1], 1)))
            node_unit = signature / np.maximum(np.linalg.norm(signature, axis=1, keepdims=True), 1.0e-12)
            relation = 0.5 * (np.clip(node_unit @ node_unit.T, -1.0, 1.0) + 1.0)
            padded_relation = np.zeros((self.max_dof, self.max_dof), dtype=np.float32)
            padded_relation[:nodes, :nodes] = relation
            sample["modal_relation_prior"] = torch.from_numpy(padded_relation)
        if self.parameter_context_length > 0:
            context_length = min(self.parameter_context_length, len(record.time))
            context_indices = np.linspace(0, len(record.time) - 1, context_length, dtype=np.int64)
            parameter_context = full_measurement[context_indices] * mask[None, :]
            sample["parameter_context"] = torch.from_numpy(parameter_context.astype(np.float32))
        if self.modal_context_length > 0:
            context_length = min(self.modal_context_length, len(record.time))
            context_indices = np.linspace(0, len(record.time) - 1, context_length, dtype=np.int64)
            modal_context = full_measurement[context_indices] * mask[None, :]
            sample["modal_context"] = torch.from_numpy(modal_context.astype(np.float32))
            frequency, modal_damping, shape = modal_targets(
                record.mass, record.stiffness, record.damping, self.num_modal_modes
            )
            mode_count = len(frequency)
            modal_frequency = np.zeros(self.num_modal_modes, dtype=np.float32)
            modal_ratio = np.zeros(self.num_modal_modes, dtype=np.float32)
            modal_shape = np.zeros((self.max_dof, self.num_modal_modes), dtype=np.float32)
            modal_valid = np.zeros(self.num_modal_modes, dtype=np.float32)
            modal_frequency[:mode_count] = frequency
            modal_ratio[:mode_count] = modal_damping
            modal_shape[:nodes, :mode_count] = shape
            modal_valid[:mode_count] = 1.0
            sample.update({
                "modal_frequency": torch.from_numpy(modal_frequency),
                "modal_damping": torch.from_numpy(modal_ratio),
                "modal_shape": torch.from_numpy(modal_shape),
                "modal_valid": torch.from_numpy(modal_valid),
            })
        return sample
