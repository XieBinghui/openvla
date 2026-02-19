"""
data_utils.py

Additional RLDS-specific data utilities.
"""

import hashlib
import json
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import dlimp as dl
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from prismatic.overwatch import initialize_overwatch

import time
import torch.distributed as dist

def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _is_dist() else 0

def _world():
    return dist.get_world_size() if _is_dist() else 1

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def tree_map(fn: Callable, tree: Dict) -> Dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: Dict) -> Dict:
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    elif tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    else:
        raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# === State / Action Processing Primitives ===


# ruff: noqa: B023
def normalize_action_and_proprio(traj: Dict, metadata: Dict, normalization_type: NormalizationType):
    """Normalizes the action and proprio fields of a trajectory using the given metadata."""
    keys_to_normalize = {"action": "action", "proprio": "observation/proprio"}

    if normalization_type == NormalizationType.NORMAL:
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x),
            )

        return traj

    elif normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
        for key, traj_key in keys_to_normalize.items():
            if normalization_type == NormalizationType.BOUNDS:
                low = metadata[key]["min"]
                high = metadata[key]["max"]
            elif normalization_type == NormalizationType.BOUNDS_Q99:
                low = metadata[key]["q01"]
                high = metadata[key]["q99"]
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1),
                    x,
                ),
            )

            # Note (Moo Jin): Map unused action dimensions (i.e., dimensions where min == max) to all 0s.
            zeros_mask = metadata[key]["min"] == metadata[key]["max"]
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == traj_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )

        return traj

    raise ValueError(f"Unknown Normalization Type {normalization_type}")


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions


# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]) -> None:
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def get_dataset_statistics(
    dataset: "dl.DLataset",
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Either computes the statistics of a dataset or loads them from a cache file if this function has been called before
    with the same `hash_dependencies`.

    Rank0-only behavior (for torchrun/DDP):
      - rank0 computes + writes cache
      - other ranks wait for cache file then load it

    Statistics include min/max/mean/std/quantiles of actions & proprio, plus transition/trajectory counts.
    """
    # -------------------------
    # helpers for distributed
    # -------------------------
    def dist_is_initialized() -> bool:
        return dist is not None and dist.is_available() and dist.is_initialized()

    def get_rank() -> int:
        return dist.get_rank() if dist_is_initialized() else 0

    def barrier() -> None:
        if dist_is_initialized():
            dist.barrier()

    rank = get_rank()

    # -------------------------
    # resolve cache paths
    # -------------------------
    unique_hash = hashlib.sha256(
        "".join(hash_dependencies).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()

    # Fallback local path for when data_dir is not writable or not provided
    local_path = os.path.expanduser(
        os.path.join("~", ".cache", "orca", f"dataset_statistics_{unique_hash}.json")
    )

    if save_dir is not None:
        path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json")
    else:
        path = local_path

    # -------------------------
    # fast path: load cache
    # -------------------------
    def load_from_path(p: str) -> Dict:
        overwatch.info(f"Loading existing dataset statistics from {p}.")
        with tf.io.gfile.GFile(p, "r") as f:
            return json.load(f)

    # First try shared path (gfile) then local_path (posix)
    if tf.io.gfile.exists(path):
        return load_from_path(path)

    if os.path.exists(local_path):
        overwatch.info(f"Loading existing dataset statistics from {local_path}.")
        with open(local_path, "r") as f:
            return json.load(f)

    # -------------------------
    # DDP: non-rank0 waits
    # -------------------------
    if dist_is_initialized() and rank != 0:
        overwatch.info(
            f"[rank {rank}] Dataset statistics cache not found yet. Waiting for rank0 to compute it..."
        )

        # Prefer waiting for the shared `path` (save_dir). If save_dir is None, `path==local_path`.
        # We also check local_path as a fallback, but ideally all ranks see `path`.
        while True:
            if tf.io.gfile.exists(path):
                break
            if os.path.exists(local_path):
                # This might help in single-node setups even if save_dir is unwritable
                break
            time.sleep(2.0)

        barrier()

        if tf.io.gfile.exists(path):
            return load_from_path(path)

        # fallback
        overwatch.info(f"[rank {rank}] Loading existing dataset statistics from {local_path}.")
        with open(local_path, "r") as f:
            return json.load(f)

    # -------------------------
    # rank0 computes statistics
    # -------------------------
    if dist_is_initialized():
        overwatch.info(f"[rank {rank}] Computing dataset statistics (rank0-only).")

    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            "proprio": (
                traj["observation"]["proprio"]
                if "proprio" in traj["observation"]
                else tf.zeros_like(traj["action"])
            ),
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    overwatch.info(
        "Computing dataset statistics. This may take a bit, but should only need to happen once."
    )

    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    for traj in tqdm(
        dataset.iterator(),
        total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
    ):
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1

    actions = np.concatenate(actions)
    proprios = np.concatenate(proprios)

    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        },
        "proprio": {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
            "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
            "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    # -------------------------
    # write cache (atomic)
    # -------------------------
    def atomic_write_json_gfile(dst: str, obj: Dict) -> None:
        tmp = f"{dst}.tmp.rank{rank}.pid{os.getpid()}"
        with tf.io.gfile.GFile(tmp, "w") as f:
            json.dump(obj, f)
        tf.io.gfile.rename(tmp, dst, overwrite=True)

    wrote_shared = False
    try:
        # Try writing to shared path first
        atomic_write_json_gfile(path, metadata)
        wrote_shared = True
        overwatch.info(f"Wrote dataset statistics to {path}.")
    except tf.errors.PermissionDeniedError:
        overwatch.warning(
            f"Could not write dataset statistics to {path} (permission denied). "
            f"Writing to {local_path} instead."
        )
    except Exception as e:
        overwatch.warning(
            f"Could not write dataset statistics to {path} (error: {type(e).__name__}: {e}). "
            f"Writing to {local_path} instead."
        )

    if not wrote_shared:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # local atomic write
        tmp = f"{local_path}.tmp.rank{rank}.pid{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(metadata, f)
        os.replace(tmp, local_path)
        overwatch.info(f"Wrote dataset statistics to {local_path}.")

    barrier()
    return metadata


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    overwatch.info(f"Saved dataset statistics file at path {out_path}")


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """
    Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation
