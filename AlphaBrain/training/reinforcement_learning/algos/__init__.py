"""RL algorithms.

Each subpackage implements one method on top of the shared `envs/` + `common/`
infrastructure. Currently:

- `RLT` — RL Token (Physical Intelligence, off-policy TD3 with frozen VLA)

Future siblings (GRPO, PPO, …) drop in here without touching any other dir.
"""
from AlphaBrain.training.reinforcement_learning.algos import RLT

__all__ = ["RLT"]
