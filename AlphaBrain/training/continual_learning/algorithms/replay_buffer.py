"""
replay_buffer.py

Experience Replay buffer for continual learning.
Stores samples from previously learned tasks and provides mixed batches
to mitigate catastrophic forgetting.

Supports:
- Reservoir sampling for memory-efficient storage
- Per-task buffer management
- Configurable replay ratio for batch mixing
- Conforms to the `CLAlgorithm` interface.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from AlphaBrain.training.continual_learning.algorithms.base import CLAlgorithm

logger = logging.getLogger(__name__)


class ReplayBuffer(CLAlgorithm):
    """Experience Replay buffer that stores samples from past tasks.

    Uses reservoir sampling to maintain a fixed-size buffer per task.
    During training, samples from the buffer are mixed with current task data
    at a configurable ratio.

    ER is a task-end-populated algorithm (the buffer is filled once after each
    task finishes via :meth:`populate_from_dataset`), so the per-step
    :meth:`observe` hook is a no-op.  The trainer calls
    :meth:`populate_from_dataset` directly in its task-end handler.

    Usage:
        buffer = ReplayBuffer(buffer_size_per_task=500)

        # After finishing task 0:
        buffer.populate_from_dataset(task_id=0, dataset=task0_dataset)

        # During task 1 training:
        replay_samples = buffer.sample(batch_size=4)  # list[dict]
    """

    def __init__(self, buffer_size_per_task: int = 500, seed: int = 42):
        """
        Args:
            buffer_size_per_task: Maximum number of samples stored per task.
            seed: Random seed for reproducibility.
        """
        self.buffer_size_per_task = buffer_size_per_task
        self.seed = seed
        self.rng = random.Random(seed)

        # task_id -> list of stored samples
        self._buffers: Dict[int, List[dict]] = {}
        # Track total samples across all tasks
        self._total_samples = 0

    @property
    def num_tasks(self) -> int:
        """Number of tasks stored in the buffer."""
        return len(self._buffers)

    @property
    def total_samples(self) -> int:
        """Total number of samples across all tasks."""
        return self._total_samples

    def is_empty(self) -> bool:
        return self._total_samples == 0

    def populate_from_dataset(self, task_id: int, dataset: Dataset, num_samples: Optional[int] = None):
        """Store samples from a dataset into the buffer using reservoir sampling.

        Args:
            task_id: Identifier for the task.
            dataset: Dataset to sample from (must support __len__ and __getitem__).
            num_samples: Number of samples to store. Defaults to buffer_size_per_task.
        """
        if num_samples is None:
            num_samples = self.buffer_size_per_task

        n = len(dataset)
        k = min(num_samples, n)

        # Reservoir sampling: select k items from n uniformly at random
        indices = list(range(n))
        self.rng.shuffle(indices)
        selected_indices = sorted(indices[:k])

        samples = []
        for idx in selected_indices:
            try:
                sample = dataset[idx]
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to read sample {idx} for task {task_id}: {e}")
                continue

        # Update buffer
        if task_id in self._buffers:
            self._total_samples -= len(self._buffers[task_id])
        self._buffers[task_id] = samples
        self._total_samples += len(samples)

        logger.info(
            f"Replay buffer: stored {len(samples)} samples for task {task_id} "
            f"(total: {self._total_samples} across {self.num_tasks} tasks)"
        )

    def sample(self, batch_size: int) -> List[dict]:
        """Sample a batch uniformly from all stored tasks.

        Args:
            batch_size: Number of samples to return.

        Returns:
            List of sample dicts. Empty list if buffer is empty.
        """
        if self.is_empty():
            return []

        # Collect all samples across tasks
        all_samples = []
        for task_samples in self._buffers.values():
            all_samples.extend(task_samples)

        # Sample without replacement when possible to maximize diversity
        if batch_size <= len(all_samples):
            return self.rng.sample(all_samples, k=batch_size)
        else:
            return self.rng.choices(all_samples, k=batch_size)

    def sample_balanced(self, batch_size: int) -> List[dict]:
        """Sample a batch with equal representation from each stored task.

        Args:
            batch_size: Number of samples to return.

        Returns:
            List of sample dicts. Empty list if buffer is empty.
        """
        if self.is_empty():
            return []

        samples_per_task = max(1, batch_size // self.num_tasks)
        result = []

        for task_samples in self._buffers.values():
            k = min(samples_per_task, len(task_samples))
            result.extend(self.rng.choices(task_samples, k=k))

        # If we need more samples to reach batch_size, sample randomly
        while len(result) < batch_size:
            task_id = self.rng.choice(list(self._buffers.keys()))
            result.append(self.rng.choice(self._buffers[task_id]))

        return result[:batch_size]

    def get_task_ids(self) -> List[int]:
        """Return list of task IDs stored in the buffer."""
        return sorted(self._buffers.keys())

    def get_task_size(self, task_id: int) -> int:
        """Return number of samples stored for a specific task."""
        return len(self._buffers.get(task_id, []))

    def clear(self):
        """Clear all stored samples."""
        self._buffers.clear()
        self._total_samples = 0

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing.

        Note: only metadata is serialized — the actual sample tensors are not
        saved (they can be large).  On resume, callers must re-populate the
        buffer by iterating each task's dataset again.
        """
        return {
            "algorithm": self.name,
            "buffer_size_per_task": self.buffer_size_per_task,
            "seed": self.seed,
            "num_tasks": self.num_tasks,
            "total_samples": self._total_samples,
            "task_sizes": {k: len(v) for k, v in self._buffers.items()},
        }

    # ------------------------------------------------------------------
    # CLAlgorithm interface
    # ------------------------------------------------------------------
    def observe(self, batch: dict, task_id: int) -> None:
        """No-op: ER populates from the full dataset at task-end, not per step.

        See :meth:`populate_from_dataset` for the actual memory update, which
        the trainer invokes from its task-end handler.
        """
        return None

    def on_task_end(self, task_id: int) -> None:
        """No-op: the trainer calls :meth:`populate_from_dataset` directly.

        (ER needs the full task dataset object, which a generic no-arg hook
        cannot provide — so the trainer orchestrates the population explicitly.)
        """
        return None

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore metadata from a snapshot produced by :meth:`state_dict`.

        Only hyperparameters are restored (buffer size, seed).  Actual samples
        must be re-populated from the task datasets on resume.
        """
        self.buffer_size_per_task = state.get(
            "buffer_size_per_task", self.buffer_size_per_task
        )
        self.seed = state.get("seed", self.seed)
        self.rng = random.Random(self.seed)
        self._buffers = {}
        self._total_samples = 0

    def __repr__(self) -> str:
        task_info = ", ".join(
            f"task_{k}={len(v)}" for k, v in sorted(self._buffers.items())
        )
        return (
            f"ReplayBuffer(buffer_size_per_task={self.buffer_size_per_task}, "
            f"tasks={self.num_tasks}, total={self._total_samples}, [{task_info}])"
        )
