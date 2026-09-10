from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


def admissible_sensor_layouts(num_nodes: int, sensor_count: int) -> list[tuple[int, ...]]:
    """Engineering-admissible layouts without optimizing placement quality."""
    layouts = []
    for layout in itertools.combinations(range(num_nodes), sensor_count):
        if sensor_count == 1:
            admissible = True
        elif sensor_count == 2:
            admissible = layout[-1] - layout[0] >= int(np.ceil(0.4 * (num_nodes - 1)))
        else:
            admissible = (
                any(index <= 2 for index in layout)
                and any(2 <= index <= 5 for index in layout)
                and any(index >= 5 for index in layout)
            )
        if admissible:
            layouts.append(layout)
    return layouts


def sensor_count_layout_split(seed: int, num_nodes: int = 8) -> dict[int, dict[str, list[tuple[int, ...]]]]:
    """Deterministic count-wise train/validation/test layout split."""
    rng = np.random.default_rng(seed)
    targets = {1: (0, 4, 4), 2: (9, 3, 3), 3: (20, 6, 6), 4: (18, 6, 6), 5: (18, 6, 6)}
    result = {}
    for count, sizes in targets.items():
        pool = admissible_sensor_layouts(num_nodes, count)
        order = rng.permutation(len(pool))
        required = sum(sizes)
        selected = [pool[int(index)] for index in order[:required]]
        train_end = sizes[0]
        val_end = train_end + sizes[1]
        result[count] = {
            "train": selected[:train_end],
            "val": selected[train_end:val_end],
            "test": selected[val_end:],
        }
    return result


def balanced_training_layouts(
    split: dict[int, dict[str, list[tuple[int, ...]]]], counts: tuple[int, ...] = (2, 3, 4, 5)
) -> list[tuple[int, ...]]:
    """Duplicate positions so uniform layout sampling implies uniform count sampling."""
    target = max(len(split[count]["train"]) for count in counts)
    balanced = []
    for count in counts:
        pool = split[count]["train"]
        balanced.extend((pool * ((target + len(pool) - 1) // len(pool)))[:target])
    return balanced


def split_audit(split: dict[int, dict[str, list[tuple[int, ...]]]]) -> dict[str, object]:
    rows, passed = {}, True
    for count, pools in split.items():
        train, val, test = map(set, (pools["train"], pools["val"], pools["test"]))
        overlap = {
            "train_val": sorted(train & val), "train_test": sorted(train & test),
            "val_test": sorted(val & test),
        }
        admissible = set(admissible_sensor_layouts(8, count))
        valid = not any(overlap.values()) and (train | val | test) <= admissible
        passed &= valid
        rows[str(count)] = {
            "pool_sizes": {name: len(values) for name, values in pools.items()},
            "overlap": overlap, "admissibility_passed": valid,
        }
    return {"by_count": rows, "passed": bool(passed), "heldout_test_accessed": False}


def balanced_count_histogram(layouts: list[tuple[int, ...]]) -> dict[int, int]:
    return dict(sorted(Counter(map(len, layouts)).items()))

