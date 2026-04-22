"""Abstract base class for continual-learning algorithms.

All CL algorithms (Experience Replay / EWC / LwF / SI / GEM / ...) implement
this interface.  The continual trainer (`AlphaBrain.training.continual_learning.train`)
only talks to the algorithm through this protocol, so new methods can be
plugged in without touching the training loop.

Current implementations
-----------------------
- `ReplayBuffer`  (algorithms.replay_buffer)
  Reservoir-sampled experience replay with uniform / balanced strategies.

Planned implementations
-----------------------
- `EWC`   Elastic Weight Consolidation (Kirkpatrick et al. 2017)
- `LwF`   Learning without Forgetting  (Li & Hoiem 2017)
- `SI`    Synaptic Intelligence         (Zenke et al. 2017)
- `GEM`   Gradient Episodic Memory      (Lopez-Paz & Ranzato 2017)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CLAlgorithm(ABC):
    """Interface every continual-learning algorithm must satisfy."""

    # ------------------------------------------------------------------
    # Training-loop hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def observe(self, batch: dict, task_id: int) -> None:
        """Called every training step with the **current task** batch.

        Replay-based methods (ER, GEM) use this to grow/refresh memory.
        Regularization methods (EWC, SI) use it to accumulate importance
        statistics (Fisher information, path integrals, etc.).
        """

    @abstractmethod
    def sample(self, batch_size: int) -> Any:
        """Return the algorithm's auxiliary artifact for this step (or None/empty to skip).

        Return shape is algorithm-specific:
          * ER / GEM : `list[dict]`       — raw samples ready to be collated.
          * EWC / SI : `dict[str, Tensor]`— per-parameter regularization terms.
          * LwF      : `dict[str, Tensor]`— teacher logits on the current batch.

        The trainer dispatches on algorithm type to combine this with the
        current-task batch (e.g. mix-in ratio for replay, KL term for LwF).
        """

    @abstractmethod
    def on_task_end(self, task_id: int) -> None:
        """Hook invoked after a task finishes.

        Typical uses:
        * EWC: snapshot parameters, compute Fisher on current task's dataset.
        * LwF: snapshot the teacher model weights.
        * ER : no-op (reservoir sampling happens online).
        """

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the algorithm state.

        This is written alongside model checkpoints so CL state survives
        interruption/restart across tasks.
        """

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore algorithm state from a dict produced by `state_dict()`."""

    # ------------------------------------------------------------------
    # Metadata (optional overrides)
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Short identifier used in logs / checkpoints (default = class name)."""
        return self.__class__.__name__
