#!/usr/bin/env python3
"""
Count RLDS/TFDS dataset sizes for OpenVLA-style datasets.

Examples:
  python vla-scripts/count_rlds_samples.py \
    --data_root_dir /path/to/datasets \
    --dataset_name bridge_orig

  python vla-scripts/count_rlds_samples.py \
    --data_root_dir /path/to/datasets \
    --dataset_name bridge_orig \
    --count_transitions
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, Tuple

import tensorflow_datasets as tfds
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count RLDS dataset samples.")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Base TFDS data root directory.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name, e.g. bridge_orig.")
    parser.add_argument(
        "--count_transitions",
        action="store_true",
        help="Iterate dataset to count total transition/step count for each split (slower).",
    )
    return parser.parse_args()


def infer_steps_in_episode(example: Dict) -> int:
    """Infer step count from one RLDS episode example."""
    if "steps" in example:
        steps = example["steps"]
        if isinstance(steps, dict) and len(steps) > 0:
            # Prefer canonical RLDS key if present.
            if "is_first" in steps:
                return len(steps["is_first"])
            # Fallback: use length of first step field.
            first_key = next(iter(steps.keys()))
            return len(steps[first_key])
        return len(steps)

    # Fallback for flattened/non-standard variants.
    if "action" in example:
        try:
            return len(example["action"])
        except TypeError:
            return 1

    return 1


def count_transitions_for_split(builder: tfds.core.DatasetBuilder, split: str) -> Tuple[int, int]:
    """Return (num_trajectories, num_transitions) by iterating split."""
    ds = builder.as_dataset(split=split, shuffle_files=False)
    num_trajectories, num_transitions = 0, 0

    for ex in tqdm(tfds.as_numpy(ds), desc=f"Counting {split}", leave=False):
        num_trajectories += 1
        num_transitions += infer_steps_in_episode(ex)

    return num_trajectories, num_transitions


def main() -> None:
    args = parse_args()

    builder = tfds.builder(args.dataset_name, data_dir=args.data_root_dir)
    info = builder.info

    print(f"Dataset      : {args.dataset_name}")
    print(f"Data root    : {args.data_root_dir}")
    print(f"TFDS version : {info.version}")
    print("")

    split_names: Iterable[str] = info.splits.keys()
    total_trajectories = 0
    total_transitions = 0

    for split_name in split_names:
        split_info = info.splits[split_name]
        traj_count = int(split_info.num_examples)
        total_trajectories += traj_count

        print(f"[{split_name}] trajectories = {traj_count}")

        if args.count_transitions:
            iter_traj, iter_transitions = count_transitions_for_split(builder, split_name)
            print(f"[{split_name}] transitions = {iter_transitions} (iterated trajectories = {iter_traj})")
            total_transitions += iter_transitions

    print("")
    print(f"Total trajectories (all splits) = {total_trajectories}")

    if args.count_transitions:
        print(f"Total transitions (all splits) = {total_transitions}")

    # Mirror OpenVLA train/val behavior for quick reference.
    if "val" in info.splits:
        print("")
        print("OpenVLA split rule: use TFDS 'train' for train and 'val' for validation.")
    elif "train" in info.splits:
        train_n = int(info.splits["train"].num_examples)
        train_95 = int(train_n * 0.95)
        val_05 = train_n - train_95
        print("")
        print("OpenVLA split rule: no 'val' split -> use train[:95%] and train[95%:].")
        print(f"Approx trajectories: train[:95%] ≈ {train_95}, train[95%:] ≈ {val_05}")


if __name__ == "__main__":
    main()
