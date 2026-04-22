"""Continual-learning algorithms.

See `AlphaBrain.training.continual_learning.algorithms.base.CLAlgorithm`
for the interface every algorithm implements.
"""
from AlphaBrain.training.continual_learning.algorithms.base import CLAlgorithm
from AlphaBrain.training.continual_learning.algorithms.replay_buffer import ReplayBuffer

__all__ = ["CLAlgorithm", "ReplayBuffer"]
