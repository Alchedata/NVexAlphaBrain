"""
Milestone 3 — SelfImprovementAgent
===================================
Autonomous orchestrator that runs the full failure → diagnosis → plan →
training → verification loop until a target KPI is reached or a stopping
condition fires.

Demo mode (simulate=True)
    Replays a precomputed 3-iteration sequence (62 → 74 → 81 → 85 %).
    Steps advance each time ``advance_step`` is called, so the React UI
    can animate at whatever speed it wants.

Real mode (simulate=False)
    Calls the existing M2 infrastructure (EvalArtifactExporter,
    PatchPlanGenerator, JobDispatcher) to run actual AlphaBrain jobs.
    Intended for customer POCs, not live investor demos.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .schemas import (
    AgentRunRequest,
    AgentRunState,
    AgentStep,
    AgentStepStatus,
    ExecutionBackend,
    FailureDiagnosis,
    LoopIteration,
    TrainingStrategy,
)

if TYPE_CHECKING:
    from .app import InMemoryStore
    from .dispatcher import JobDispatcher
    from .exporters import EvalArtifactExporter
    from .llm_narrator import LLMNarrator
    from .patch_plan_generator import PatchPlanGenerator


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _sid() -> str:
    return uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Pre-seeded demo iterations (Mode A — precomputed replay)
# ---------------------------------------------------------------------------

_DEMO_LOOPS: list[dict[str, Any]] = [
    {
        "patch_cluster": "Occlusion / object visibility",
        "patch_strategy": "continual_learning",
        "eval_before": 0.62,
        "eval_after": 0.74,
        "steps": [
            ("eval",       "Run eval",         "Triggered evaluation of ckpt_v0.7 on LIBERO Kitchen benchmark."),
            ("diagnose",   "Diagnose failures", "Identified dominant failure cluster: Occlusion / object visibility (38% of failures). "
                                                "Root causes: camera angle variation, object-behind-object cases."),
            ("plan",       "Generate plan",     "Selected continual_learning strategy via alphabrain_cl. "
                                                "Found matching recipe in Platform Memory: occlusion_recovery_v1 (used 3×, avg +9% uplift). Applying."),
            ("dispatch",   "Dispatch training", "Training job dispatched to AlphaBrain CL. "
                                                "Config: 200 patch episodes, 70% real / 30% synthetic. Estimated time: 45 min."),
            ("verify",     "Verify results",    "Re-evaluation complete. Success rate improved from 62% → 74%. Uplift: +12pp."),
            ("memory",     "Save to memory",    "Saved recipe occlusion_cl_patch_v1 to Platform Memory. "
                                                "Recipe confidence: high. Pattern fingerprint stored for future reuse."),
            ("stop_check", "Check stopping",    "Target KPI 75% not yet reached (74%). Improvement delta +12pp exceeds threshold. Continuing."),
        ],
    },
    {
        "patch_cluster": "Recovery / error correction",
        "patch_strategy": "fine_tune",
        "eval_before": 0.74,
        "eval_after": 0.81,
        "steps": [
            ("eval",       "Run eval",         "Triggered evaluation of ckpt_v1.0 (post-occlusion patch) on LIBERO Kitchen benchmark."),
            ("diagnose",   "Diagnose failures", "Primary cluster shifted to Recovery / error correction (31% of failures). "
                                                "Root cause: robot unable to self-correct after near-miss grasp failures."),
            ("plan",       "Generate plan",     "Selected fine_tune strategy on teleop correction trajectories via alphabrain_finetune. "
                                                "No prior recipe found — agent will experiment and record outcome."),
            ("dispatch",   "Dispatch training", "Fine-tune job dispatched. "
                                                "Config: 150 teleop correction clips, 80% real / 20% synthetic. Estimated time: 30 min."),
            ("verify",     "Verify results",    "Re-evaluation complete. Success rate improved from 74% → 81%. Uplift: +7pp."),
            ("memory",     "Save to memory",    "New recipe recovery_finetune_v1 saved. "
                                                "Confidence: medium (first use). Will be promoted on next successful application."),
            ("stop_check", "Check stopping",    "Target KPI 75% reached and exceeded (81%). "
                                                "Continuing one more loop to characterize diminishing-returns boundary."),
        ],
    },
    {
        "patch_cluster": "Lighting / appearance shift",
        "patch_strategy": "continual_learning",
        "eval_before": 0.81,
        "eval_after": 0.85,
        "steps": [
            ("eval",       "Run eval",         "Triggered evaluation of ckpt_v1.1 (post-recovery patch) on LIBERO Kitchen benchmark."),
            ("diagnose",   "Diagnose failures", "Remaining failures clustered around Lighting / appearance shift (18% of failures). "
                                                "Lower severity; root cause is lighting variance not seen in training distribution."),
            ("plan",       "Generate plan",     "Selected continual_learning with lighting-augmented episodes via alphabrain_cl. "
                                                "Expected uplift is smaller — this is a diminishing-returns zone."),
            ("dispatch",   "Dispatch training", "CL job dispatched with 80 lighting-augmented patch episodes. Estimated time: 20 min."),
            ("verify",     "Verify results",    "Re-evaluation complete. Success rate improved from 81% → 85%. Uplift: +4pp."),
            ("memory",     "Save to memory",    "Recipe lighting_cl_patch_v1 saved. Platform Memory now has 3 new recipes from this project."),
            ("stop_check", "Check stopping",    "Improvement delta +4pp is below the 5pp diminishing-returns threshold. "
                                                "Target KPI already exceeded. Agent terminates — convergence reached at 85%."),
        ],
    },
]


def _build_demo_run(request: AgentRunRequest) -> AgentRunState:
    """Construct a fully-precomputed AgentRunState in demo/simulate mode."""
    run_id = f"agent_{uuid4().hex[:10]}"
    iterations: list[LoopIteration] = []

    for idx, loop_def in enumerate(_DEMO_LOOPS[: request.max_iterations], start=1):
        steps: list[AgentStep] = []
        for step_type, label, message in loop_def["steps"]:
            steps.append(
                AgentStep(
                    step_id=_sid(),
                    step_type=step_type,
                    status="pending",
                    label=label,
                    message=message,
                )
            )
        iterations.append(
            LoopIteration(
                iteration_index=idx,
                patch_strategy=loop_def["patch_strategy"],
                patch_cluster=loop_def["patch_cluster"],
                eval_before=loop_def["eval_before"],
                eval_after=loop_def["eval_after"],
                steps=steps,
                status="pending",
            )
        )

    # Mark the very first step of the first iteration as "running" so the
    # UI immediately has something to show.
    if iterations and iterations[0].steps:
        iterations[0].status = "running"
        iterations[0].steps[0].status = "running"
        iterations[0].steps[0].started_at = _utc()

    return AgentRunState(
        agent_run_id=run_id,
        project_id=request.project_id,
        target_kpi=request.target_kpi,
        max_iterations=request.max_iterations,
        diminishing_returns_threshold=request.diminishing_returns_threshold,
        current_iteration=1,
        status="running",
        iterations=iterations,
        reasoning_log=[
            f"Agent started. Target KPI: {int(request.target_kpi * 100)}%. "
            f"Max iterations: {request.max_iterations}.",
            "Loading checkpoint ckpt_v0.7 from Platform Memory.",
        ],
    )


class SelfImprovementAgent:
    """
    Orchestrates the autonomous failure-to-fix loop.

    In **simulate** mode every call to ``advance_step`` moves the pre-baked
    demo state forward by exactly one step, returning the updated
    ``AgentRunState``.  The React frontend polls this and re-renders.

    In **real** mode the agent drives actual M2 infrastructure.  Each step is
    blocking, so call ``run_async`` to start it in a background thread.
    """

    def __init__(
        self,
        store: "InMemoryStore",
        dispatcher: "JobDispatcher",
        exporter: "EvalArtifactExporter",
        patch_plan_generator: "PatchPlanGenerator",
        narrator: "LLMNarrator | None" = None,
    ) -> None:
        self._store = store
        self._dispatcher = dispatcher
        self._exporter = exporter
        self._planner = patch_plan_generator
        self._narrator = narrator
        self._locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, request: AgentRunRequest) -> AgentRunState:
        """Create and persist a new agent run, then return initial state."""
        if request.simulate:
            state = _build_demo_run(request)
        else:
            state = self._start_real(request)

        self._store.agent_runs[state.agent_run_id] = state
        self._locks[state.agent_run_id] = threading.Lock()
        return state

    def advance_step(self, agent_run_id: str) -> AgentRunState:
        """
        Advance the agent by exactly one step (demo/simulate mode).

        Completes the current running step, then marks the next pending step
        as running.  When all steps in a loop are done, advances to the next
        loop iteration.  When all iterations are done (or a stop condition
        fires), marks the run as completed/stopped.
        """
        lock = self._locks.get(agent_run_id)
        if lock is None:
            # Reconstruct lock if it disappeared (e.g. server restart)
            self._locks[agent_run_id] = threading.Lock()
            lock = self._locks[agent_run_id]

        with lock:
            state = self._store.agent_runs.get(agent_run_id)
            if state is None:
                raise KeyError(f"Agent run {agent_run_id!r} not found")
            if state.status in ("completed", "stopped"):
                return state

            state = self._advance_demo(state)
            state.updated_at = _utc()
            self._store.agent_runs[agent_run_id] = state
            return state

    def get(self, agent_run_id: str) -> AgentRunState:
        state = self._store.agent_runs.get(agent_run_id)
        if state is None:
            raise KeyError(f"Agent run {agent_run_id!r} not found")
        return state

    # ------------------------------------------------------------------
    # Demo advance logic
    # ------------------------------------------------------------------

    @staticmethod
    def _advance_demo(state: AgentRunState) -> AgentRunState:
        """Mutate-and-return: mark running→completed, activate next step."""
        now = _utc()

        for loop in state.iterations:
            if loop.status not in ("running", "pending"):
                continue

            if loop.status == "pending":
                loop.status = "running"

            # Find the first running step and complete it
            for i, step in enumerate(loop.steps):
                if step.status != "running":
                    continue  # skip completed/pending steps until we hit the running one

                step.status = "completed"
                step.completed_at = now

                # Narrate completion to reasoning log
                state.reasoning_log.append(
                    f"[Loop {loop.iteration_index}] {step.label}: {step.message}"
                )

                # Activate the next step in this loop if it exists
                if i + 1 < len(loop.steps):
                    nxt = loop.steps[i + 1]
                    nxt.status = "running"
                    nxt.started_at = now
                    return state

                # No more steps — this loop is done
                loop.status = "completed"
                if loop.eval_after is not None:
                    loop_text = (
                        f"[Loop {loop.iteration_index}] Completed. "
                        f"{int(loop.eval_before * 100)}% → {int(loop.eval_after * 100)}%. "
                    )
                    # Check stopping for the last (stop_check) step
                    last_step = loop.steps[-1]
                    if last_step.step_type == "stop_check":
                        # Detect whether the stop message announces convergence
                        if "terminates" in last_step.message or "converge" in last_step.message:
                            state.status = "stopped"
                            state.stop_reason = last_step.message
                            state.reasoning_log.append(loop_text + "Agent stopped: diminishing returns / convergence.")
                            return state
                    state.reasoning_log.append(loop_text)

                # Advance to the next loop
                state.current_iteration += 1
                next_loop_idx = loop.iteration_index  # 0-based offset into list
                if next_loop_idx < len(state.iterations):
                    next_loop = state.iterations[next_loop_idx]
                    next_loop.status = "running"
                    if next_loop.steps:
                        next_loop.steps[0].status = "running"
                        next_loop.steps[0].started_at = now
                    return state

                # All loops exhausted — mark completed
                state.status = "completed"
                state.stop_reason = "All planned iterations completed."
                state.reasoning_log.append("Agent run completed successfully.")
                return state

        return state

    # ------------------------------------------------------------------
    # Real-mode skeleton (M3 → M4 extension point)
    # ------------------------------------------------------------------

    def _start_real(self, request: AgentRunRequest) -> AgentRunState:
        """
        Bootstrap a real agent run.  Full implementation is a M3→M4 extension.
        Currently creates a single-iteration run that drives the M2 pipeline.
        """
        run_id = f"agent_{uuid4().hex[:10]}"
        steps = [
            AgentStep(step_id=_sid(), step_type="eval",       status="pending", label="Run eval",          message=""),
            AgentStep(step_id=_sid(), step_type="diagnose",   status="pending", label="Diagnose failures", message=""),
            AgentStep(step_id=_sid(), step_type="plan",       status="pending", label="Generate plan",     message=""),
            AgentStep(step_id=_sid(), step_type="dispatch",   status="pending", label="Dispatch training", message=""),
            AgentStep(step_id=_sid(), step_type="verify",     status="pending", label="Verify results",    message=""),
            AgentStep(step_id=_sid(), step_type="memory",     status="pending", label="Save to memory",    message=""),
            AgentStep(step_id=_sid(), step_type="stop_check", status="pending", label="Check stopping",    message=""),
        ]
        loop = LoopIteration(
            iteration_index=1,
            patch_strategy="continual_learning",
            patch_cluster="unknown",
            eval_before=0.0,
            steps=steps,
            status="pending",
        )
        state = AgentRunState(
            agent_run_id=run_id,
            project_id=request.project_id,
            target_kpi=request.target_kpi,
            max_iterations=request.max_iterations,
            diminishing_returns_threshold=request.diminishing_returns_threshold,
            current_iteration=1,
            status="idle",
            iterations=[loop],
        )
        # Launch real execution in background
        thread = threading.Thread(
            target=self._run_real_loop,
            args=(state, request),
            daemon=True,
        )
        thread.start()
        return state

    def _run_real_loop(self, state: AgentRunState, request: AgentRunRequest) -> None:
        """Background thread for real-mode execution. Partial implementation."""
        # Placeholder — real implementation in M4
        pass

    # ------------------------------------------------------------------
    # Tool implementations (callable from real-mode loop)
    # ------------------------------------------------------------------

    def tool_diagnose_failures(self, eval_run_id: str) -> FailureDiagnosis:
        eval_run = self._store.eval_runs.get(eval_run_id)
        if eval_run is None:
            raise ValueError(f"EvalRun {eval_run_id!r} not found")

        clusters = sorted(eval_run.failure_clusters, key=lambda c: c.share_of_failures, reverse=True)
        primary = clusters[0] if clusters else None

        _STRATEGY_MAP: dict[str, tuple[TrainingStrategy, ExecutionBackend]] = {
            "occlusion":  ("continual_learning", "alphabrain_cl"),
            "recovery":   ("fine_tune",           "alphabrain_finetune"),
            "language":   ("vlm_cotrain",          "alphabrain_vlm_cotrain"),
            "lighting":   ("continual_learning",   "alphabrain_cl"),
            "temporal":   ("world_model_verification", "alphabrain_world_model"),
        }

        strategy: TrainingStrategy = "continual_learning"
        backend: ExecutionBackend = "alphabrain_cl"
        if primary:
            for keyword, (s, b) in _STRATEGY_MAP.items():
                if keyword in primary.failure_pattern.lower() or keyword in primary.label.lower():
                    strategy, backend = s, b
                    break

        narrator = self._narrator
        reasoning = (
            narrator.narrate_diagnosis(primary, eval_run) if narrator and primary
            else (
                f"Primary failure cluster '{primary.label}' accounts for "
                f"{int(primary.share_of_failures * 100)}% of failures. "
                f"Recommended strategy: {strategy}."
                if primary
                else "No failure clusters detected."
            )
        )

        return FailureDiagnosis(
            primary_cluster_id=primary.cluster_id if primary else "none",
            primary_cluster_label=primary.label if primary else "None",
            root_causes=[c.failure_pattern for c in clusters[:3]],
            recommended_strategy=strategy,
            recommended_backend=backend,
            reasoning=reasoning,
            confidence=primary.share_of_failures if primary else 0.0,
        )
