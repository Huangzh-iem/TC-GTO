from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


def stratified_three_sensor_layouts(num_floors: int = 8) -> list[tuple[int, int, int]]:
    """Enumerate unique low/mid/high three-sensor layouts (zero based)."""

    if num_floors != 8:
        raise ValueError("the preregistered C1 layout protocol is defined for 8 floors")
    low = range(0, 3)   # floors 1--3
    middle = range(2, 6)  # floors 3--6
    high = range(5, 8)  # floors 6--8
    layouts = {
        tuple(sorted((i, j, k)))
        for i, j, k in itertools.product(low, middle, high)
        if len({i, j, k}) == 3
    }
    return sorted(layouts)


def split_sensor_layouts(
    layouts: Iterable[tuple[int, int, int]],
    seed: int,
    *,
    validation_anchor: tuple[int, int, int] = (0, 2, 7),
) -> dict[str, list[tuple[int, int, int]]]:
    """Deterministic 60/20/20 split with the known 1/3/8 failure in validation."""

    unique = sorted(set(tuple(layout) for layout in layouts))
    if validation_anchor not in unique:
        raise ValueError("validation anchor must be a valid enumerated layout")
    rng = np.random.default_rng(seed)
    shuffled = [unique[index] for index in rng.permutation(len(unique))]
    n_train = int(round(0.60 * len(shuffled)))
    n_val = int(round(0.20 * len(shuffled)))
    split = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
    if validation_anchor not in split["val"]:
        source = next(name for name in ("train", "test") if validation_anchor in split[name])
        replacement = split["val"][0]
        split[source][split[source].index(validation_anchor)] = replacement
        split["val"][0] = validation_anchor
    validate_layout_split(split)
    return {name: sorted(values) for name, values in split.items()}


def validate_layout_split(split: dict[str, list[tuple[int, int, int]]]) -> None:
    required = {"train", "val", "test"}
    if set(split) != required:
        raise ValueError(f"layout split must have keys {sorted(required)}")
    sets = {name: set(values) for name, values in split.items()}
    if any(sets[left] & sets[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("sensor layout split leakage detected")
    all_layouts = set().union(*sets.values())
    if sum(len(values) for values in sets.values()) != len(all_layouts):
        raise ValueError("duplicate layout within a split")
    for layout in all_layouts:
        if len(layout) != 3 or len(set(layout)) != 3:
            raise ValueError("every C1 layout must contain exactly three unique floors")

