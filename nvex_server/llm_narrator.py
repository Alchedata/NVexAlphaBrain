"""
LLM Narrator — Milestone 3D
============================
Generates natural-language explanations for agent steps.

- If ``OPENAI_API_KEY`` is set in the environment, uses the OpenAI chat
  completions API (gpt-4o-mini by default — fast and cheap).
- Otherwise falls back to deterministic template strings so the demo works
  with no external dependencies.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schemas import EvalRun, FailureCluster


_SYSTEM_PROMPT = """\
You are Nvex, an AI robot-policy improvement platform.
Explain each step of an autonomous improvement loop in 2 sentences max.
Be concrete: mention cluster names, percentages, and strategy names.
Write in present tense and first-person plural ("We identified…").
"""

_TEMPLATES = {
    "eval": "Triggered evaluation of {checkpoint} on {benchmark}. Waiting for results.",
    "diagnose": (
        "Identified dominant failure cluster: {label} ({pct}% of failures). "
        "Root cause: {failure_pattern}."
    ),
    "plan": (
        "Selected {strategy} strategy via {backend}. "
        "Targeting {episodes} patch episodes at {ratio}% real data."
    ),
    "dispatch": (
        "Training job dispatched to AlphaBrain. "
        "Config: {episodes} patch episodes, {ratio}% real / {synth}% synthetic."
    ),
    "verify": "Re-evaluation complete. Success rate: {before}% → {after}%. Uplift: +{delta}pp.",
    "memory": "Saved recipe {name} to Platform Memory. Confidence: {conf}.",
    "stop_check": "Improvement delta +{delta}pp vs. threshold {threshold}pp. {decision}.",
}


class LLMNarrator:
    """
    Narrates agent reasoning steps in plain English.

    If ``openai`` is importable and ``OPENAI_API_KEY`` is set, each
    ``narrate_*`` call makes a real LLM request.  Otherwise templates are
    used silently — no exception is raised.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model
        self._client: Any = None
        self._init_openai()

    def _init_openai(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            return
        try:
            import openai  # type: ignore[import-untyped]

            self._client = openai.OpenAI()
        except ImportError:
            pass

    def _llm(self, user_prompt: str) -> str:
        if self._client is None:
            return ""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=120,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------
    # Public narration methods
    # ------------------------------------------------------------------

    def narrate_diagnosis(self, cluster: "FailureCluster", eval_run: "EvalRun") -> str:
        tmpl = _TEMPLATES["diagnose"].format(
            label=cluster.label,
            pct=int(cluster.share_of_failures * 100),
            failure_pattern=cluster.failure_pattern,
        )
        if self._client is None:
            return tmpl
        prompt = (
            f"The dominant failure cluster is '{cluster.label}' "
            f"({int(cluster.share_of_failures * 100)}% of failures). "
            f"Failure pattern: {cluster.failure_pattern}. "
            f"Affected tasks: {', '.join(cluster.affected_tasks[:3])}. "
            "Explain what this means and what to do about it."
        )
        llm_text = self._llm(prompt)
        return llm_text or tmpl

    def narrate_plan(
        self,
        strategy: str,
        backend: str,
        episodes: int,
        real_ratio: float,
        memory_recipe: str | None,
    ) -> str:
        tmpl = _TEMPLATES["plan"].format(
            strategy=strategy,
            backend=backend,
            episodes=episodes,
            ratio=int(real_ratio * 100),
        )
        if self._client is None:
            return tmpl
        recipe_note = (
            f"A prior recipe '{memory_recipe}' was found in Platform Memory."
            if memory_recipe
            else "No prior recipe found — agent will experiment."
        )
        prompt = (
            f"We selected '{strategy}' via '{backend}' targeting {episodes} patch episodes "
            f"({int(real_ratio * 100)}% real data). {recipe_note} "
            "Narrate this planning step."
        )
        llm_text = self._llm(prompt)
        return llm_text or tmpl

    def narrate_verify(self, before: float, after: float, threshold: float) -> str:
        delta = round(after - before, 3)
        tmpl = _TEMPLATES["verify"].format(
            before=int(before * 100),
            after=int(after * 100),
            delta=int(delta * 100),
        )
        if self._client is None:
            return tmpl
        prompt = (
            f"The policy improved from {int(before * 100)}% to {int(after * 100)}% "
            f"(+{int(delta * 100)}pp). The diminishing-returns threshold is {int(threshold * 100)}pp. "
            "Narrate this verification result."
        )
        llm_text = self._llm(prompt)
        return llm_text or tmpl

    def narrate_stop_check(
        self,
        delta: float,
        threshold: float,
        current_kpi: float,
        target_kpi: float,
    ) -> str:
        decision = (
            "Target exceeded — agent terminates."
            if current_kpi >= target_kpi
            else (
                "Diminishing returns detected — agent terminates."
                if delta < threshold
                else "Improvement sufficient — continuing."
            )
        )
        tmpl = _TEMPLATES["stop_check"].format(
            delta=int(delta * 100),
            threshold=int(threshold * 100),
            decision=decision,
        )
        if self._client is None:
            return tmpl
        prompt = (
            f"Current policy KPI: {int(current_kpi * 100)}%. "
            f"Target: {int(target_kpi * 100)}%. "
            f"Delta this loop: +{int(delta * 100)}pp. Threshold: {int(threshold * 100)}pp. "
            f"Decision: {decision} Narrate the stopping check."
        )
        llm_text = self._llm(prompt)
        return llm_text or tmpl
