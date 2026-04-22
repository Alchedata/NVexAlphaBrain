"""
continual_learning.py

Defines continual learning task sequences for sequential task training.
Each sequence specifies a base data_mix and task ordering.
Provides utilities to filter datasets by task_index for per-task training.
"""

import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Continual Learning Task Sequences
# ============================================================================
# Each sequence defines:
#   - base_data_mix: the data_mix name in DATASET_NAMED_MIXTURES
#   - num_tasks: number of tasks (auto-detected from dataset if not specified)
#   - task_order: optional explicit ordering of task indices (default: 0..num_tasks-1)

CL_TASK_SEQUENCES = {
    # LIBERO suites — each suite contains 10 tasks with 50 demos each
    "libero_spatial": {
        "base_data_mix": "libero_spatial",
        "num_tasks": 10,
    },
    "libero_object": {
        "base_data_mix": "libero_object",
        "num_tasks": 10,
    },
    "libero_goal": {
        "base_data_mix": "libero_goal",
        "num_tasks": 10,
    },
    "libero_long": {
        "base_data_mix": "libero_long",
        "num_tasks": 10,
    },
}


def get_task_sequence(sequence_name: str) -> dict:
    """Retrieve a CL task sequence by name."""
    if sequence_name not in CL_TASK_SEQUENCES:
        raise ValueError(
            f"Unknown CL task sequence: {sequence_name}. "
            f"Available: {list(CL_TASK_SEQUENCES.keys())}"
        )
    return CL_TASK_SEQUENCES[sequence_name]


# ============================================================================
# Episode-to-Task Mapping
# ============================================================================

def build_episode_task_map(dataset) -> Dict[int, List[int]]:
    """Build mapping from task_index to list of episode_ids by reading episode data.

    Args:
        dataset: A LeRobotSingleDataset instance.

    Returns:
        Dict mapping task_index -> list of trajectory_ids (episode indices).
    """
    task_to_episodes: Dict[int, List[int]] = defaultdict(list)
    seen_episodes = set()

    for traj_id in dataset.trajectory_ids:
        if traj_id in seen_episodes:
            continue
        seen_episodes.add(traj_id)

        try:
            data = dataset.get_trajectory_data(traj_id)
            if "task_index" in data.columns:
                task_idx = int(data["task_index"].iloc[0])
            else:
                # Fallback: try annotation-based task index
                annotation_cols = [c for c in data.columns if "task" in c.lower()]
                if annotation_cols:
                    task_idx = int(data[annotation_cols[0]].iloc[0])
                else:
                    logger.warning(
                        f"No task_index column found for episode {traj_id}, assigning task 0"
                    )
                    task_idx = 0
            task_to_episodes[task_idx].append(traj_id)
        except Exception as e:
            logger.warning(f"Failed to read task_index for episode {traj_id}: {e}")
            continue

    # Clear dataset cache
    dataset.curr_traj_data = None
    dataset.curr_traj_id = None

    logger.info(
        f"Built episode-task map: {len(task_to_episodes)} tasks, "
        f"{sum(len(v) for v in task_to_episodes.values())} total episodes"
    )
    for task_idx in sorted(task_to_episodes.keys()):
        logger.info(f"  Task {task_idx}: {len(task_to_episodes[task_idx])} episodes")

    return dict(task_to_episodes)


# ============================================================================
# Task-Filtered Dataset Wrapper
# ============================================================================

class TaskFilteredDataset(Dataset):
    """Wraps a LeRobotMixtureDataset to only expose steps from specific task indices.

    This is a lightweight wrapper that filters the base dataset's step sampling
    without copying data or modifying the underlying dataset.
    """

    def __init__(self, base_dataset, task_indices: List[int], episode_task_map: Dict[int, List[int]]):
        """
        Args:
            base_dataset: A LeRobotMixtureDataset (or LeRobotSingleDataset).
            task_indices: List of task_index values to include.
            episode_task_map: Mapping from task_index -> list of episode_ids.
        """
        self.base_dataset = base_dataset
        self.task_indices = task_indices

        # Build set of valid episode ids for fast lookup
        self.valid_episodes = set()
        for ti in task_indices:
            if ti in episode_task_map:
                self.valid_episodes.update(episode_task_map[ti])

        # For MixtureDataset: filter each sub-dataset's all_steps
        # For SingleDataset: filter directly
        if hasattr(base_dataset, 'datasets'):
            # MixtureDataset
            self._filtered_steps_per_dataset = []
            self._total_steps = 0
            for ds in base_dataset.datasets:
                filtered = [
                    (traj_id, base_idx)
                    for traj_id, base_idx in ds.all_steps
                    if traj_id in self.valid_episodes
                ]
                self._filtered_steps_per_dataset.append(filtered)
                self._total_steps += len(filtered)
        else:
            # SingleDataset
            self._filtered_steps = [
                (traj_id, base_idx)
                for traj_id, base_idx in base_dataset.all_steps
                if traj_id in self.valid_episodes
            ]
            self._total_steps = len(self._filtered_steps)

        logger.info(
            f"TaskFilteredDataset: tasks={task_indices}, "
            f"episodes={len(self.valid_episodes)}, steps={self._total_steps}"
        )

    def __len__(self) -> int:
        return self._total_steps

    def __getitem__(self, index: int) -> dict:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if hasattr(self.base_dataset, 'datasets'):
                    return self._getitem_mixture(index)
                else:
                    return self._getitem_single(index)
            except Exception as e:
                import random
                logger.warning(
                    f"[TaskFilteredDataset] Error loading index {index} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                index = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"[TaskFilteredDataset] Failed to load data after {max_retries} retries"
        )

    def _getitem_single(self, index: int) -> dict:
        """Get item from filtered single dataset."""
        traj_id, base_idx = self._filtered_steps[index % len(self._filtered_steps)]
        ds = self.base_dataset
        raw_data = ds.get_step_data(traj_id, base_idx)
        data = ds.transforms(raw_data)
        sample = ds._pack_sample(data)
        if hasattr(ds, 'tag'):
            sample["robot_tag"] = ds.tag
        return sample

    def _getitem_mixture(self, index: int) -> dict:
        """Get item from filtered mixture dataset with weighted sampling."""
        import random
        # Weighted random selection across sub-datasets
        dataset_weights = []
        for i, steps in enumerate(self._filtered_steps_per_dataset):
            if len(steps) > 0:
                dataset_weights.append((i, len(steps)))

        if not dataset_weights:
            raise ValueError("No valid steps in filtered dataset")

        total = sum(w for _, w in dataset_weights)
        r = random.random() * total
        cumulative = 0
        ds_idx = dataset_weights[0][0]
        for idx, w in dataset_weights:
            cumulative += w
            if r <= cumulative:
                ds_idx = idx
                break

        steps = self._filtered_steps_per_dataset[ds_idx]
        step_idx = random.randint(0, len(steps) - 1)
        traj_id, base_idx = steps[step_idx]

        ds = self.base_dataset.datasets[ds_idx]
        raw_data = ds.get_step_data(traj_id, base_idx)
        data = ds.transforms(raw_data)
        sample = ds._pack_sample(data)
        sample["robot_tag"] = ds.tag
        return sample

    @property
    def datasets(self):
        """Expose underlying datasets for compatibility."""
        if hasattr(self.base_dataset, 'datasets'):
            return self.base_dataset.datasets
        return [self.base_dataset]

    def save_dataset_statistics(self, path):
        """Delegate to base dataset."""
        if hasattr(self.base_dataset, 'save_dataset_statistics'):
            self.base_dataset.save_dataset_statistics(path)
