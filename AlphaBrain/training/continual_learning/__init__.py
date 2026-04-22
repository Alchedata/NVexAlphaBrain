"""Continual Learning module.

Sub-packages:
    algorithms/   — CL algorithms (ReplayBuffer, …) and their `CLAlgorithm` base.
    datasets/     — Task sequences and per-task dataset filtering.

Top-level entry:
    train         — Continual training loop (`AlphaBrain.training.continual_learning.train.main`).

Re-exports for backward compatibility (old import paths still work):
    `ReplayBuffer`  ← `algorithms.replay_buffer.ReplayBuffer`
"""
from AlphaBrain.training.continual_learning.algorithms import CLAlgorithm, ReplayBuffer

__all__ = ["CLAlgorithm", "ReplayBuffer"]
