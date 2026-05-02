from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentStepType = Literal["eval", "diagnose", "plan", "dispatch", "verify", "memory", "stop_check"]
AgentStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
AgentRunStatus = Literal["idle", "running", "completed", "stopped"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


ExecutionBackend = Literal[
    "alphabrain_cl",
    "alphabrain_finetune",
    "alphabrain_eval",
    "alphabrain_vlm_cotrain",
    "alphabrain_world_model",
]

JobStatus = Literal["queued", "running", "completed", "failed"]
Severity = Literal["low", "medium", "high", "critical"]
TrainingStrategy = Literal["continual_learning", "fine_tune", "vlm_cotrain", "world_model_verification"]
ArtifactType = Literal["auto", "generic_json", "libero_eval_json", "robocasa365_aggregate", "robocasa_tabletop_stats", "libero_log"]


class ArtifactBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    videos: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    metrics_json: str | None = None
    source_path: str | None = None


class TaskBreakdownEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_name: str
    success_rate: float = Field(ge=0.0, le=1.0)
    attempts: int | None = Field(default=None, ge=0)
    successes: int | None = Field(default=None, ge=0)


class FailureCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    label: str
    failure_pattern: str
    affected_tasks: list[str] = Field(default_factory=list)
    share_of_failures: float = Field(ge=0.0, le=1.0)
    failure_count: int = Field(ge=0)
    severity: Severity = "medium"


class EvalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    project_id: str
    benchmark_suite: str
    checkpoint: str | None = None
    overall_success: float = Field(ge=0.0, le=1.0)
    task_breakdown: list[TaskBreakdownEntry] = Field(default_factory=list)
    failure_clusters: list[FailureCluster] = Field(default_factory=list)
    artifacts: ArtifactBundle = Field(default_factory=ArtifactBundle)
    created_at: datetime = Field(default_factory=utc_now)


class TargetDataSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch_episodes: int = Field(ge=0)
    teleop_corrections: int = Field(ge=0)
    lighting_variants: int = Field(default=0, ge=0)
    language_augmentations: int = Field(default=0, ge=0)


class SourceRatio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    real: float = Field(ge=0.0, le=1.0)
    synthetic: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ensure_total_is_one(self) -> "SourceRatio":
        total = round(self.real + self.synthetic, 6)
        if total != 1.0:
            raise ValueError("source_ratio.real + source_ratio.synthetic must equal 1.0")
        return self


class PatchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    project_id: str
    based_on_eval_run: str
    root_causes: list[str] = Field(default_factory=list)
    target_data_spec: TargetDataSpec
    annotation_schema: str
    source_ratio: SourceRatio
    training_strategy: TrainingStrategy
    execution_backend: ExecutionBackend
    verification_spec: str
    expected_uplift: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)


class IterationResultSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success_before: float = Field(ge=0.0, le=1.0)
    success_after: float = Field(ge=0.0, le=1.0)


class IterationArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logs: list[str] = Field(default_factory=list)
    videos: list[str] = Field(default_factory=list)
    eval_runs: list[str] = Field(default_factory=list)
    metadata_path: str | None = None


class IterationJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_id: str
    project_id: str
    plan_id: str
    based_on_checkpoint: str
    status: JobStatus
    execution_backend: ExecutionBackend
    config: dict[str, Any] = Field(default_factory=dict)
    command: str | None = None
    log_path: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    output_checkpoint: str | None = None
    after_eval_run_id: str | None = None
    result_summary: IterationResultSummary | None = None
    artifacts: IterationArtifacts = Field(default_factory=IterationArtifacts)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ReusableAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    type: Literal["recipe", "template", "failure_pattern", "verification_setup"]
    name: str
    source_project: str
    reuse_count: int = Field(default=0, ge=0)
    linked_iteration: str
    description: str


class ImprovementReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration_id: str
    plan_id: str
    project_id: str
    success_before: float = Field(ge=0.0, le=1.0)
    success_after: float = Field(ge=0.0, le=1.0)
    uplift: float = Field(ge=0.0, le=1.0)
    summary: str
    changes: list[str] = Field(default_factory=list)
    next_target: str | None = None
    assets_created: list[ReusableAsset] = Field(default_factory=list)


class ProjectContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    checkpoint: str
    domain: str
    suite: str
    status: str
    status_note: str
    top_risk: str
    next_action: str


class PlatformMemoryStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipes: int = Field(ge=0)
    templates: int = Field(ge=0)
    patterns: int = Field(ge=0)
    projects: int = Field(ge=0)


class PlatformMemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipes: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    stats: PlatformMemoryStats


class EvalImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_run: EvalRun | None = None
    artifact_path: str | None = None
    artifact_type: ArtifactType = "auto"
    project_id: str | None = None
    benchmark_suite: str | None = None
    checkpoint: str | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "EvalImportRequest":
        if self.eval_run is None and self.artifact_path is None:
            raise ValueError("either eval_run or artifact_path must be provided")
        if self.artifact_path is not None and not Path(self.artifact_path).exists():
            raise ValueError(f"artifact_path does not exist: {self.artifact_path}")
        return self


class PlanGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_run_id: str | None = None
    eval_run: EvalRun | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "PlanGenerationRequest":
        if not self.eval_run_id and self.eval_run is None:
            raise ValueError("either eval_run_id or eval_run must be provided")
        return self


class IterationStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    checkpoint: str
    execution_backend: ExecutionBackend | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class DemoStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectContext
    current_eval_run: EvalRun
    patch_plan: PatchPlan
    iteration_job: IterationJob
    report: ImprovementReport
    platform_memory: PlatformMemorySnapshot


# ---------------------------------------------------------------------------
# Milestone 3 — Self-Improving Agent schemas
# ---------------------------------------------------------------------------


class FailureDiagnosis(BaseModel):
    """Structured output of the FailureDiagnoser tool."""

    model_config = ConfigDict(extra="forbid")

    primary_cluster_id: str
    primary_cluster_label: str
    root_causes: list[str] = Field(default_factory=list)
    recommended_strategy: TrainingStrategy
    recommended_backend: ExecutionBackend
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class AgentStep(BaseModel):
    """A single step in the autonomous improvement loop."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    step_type: AgentStepType
    status: AgentStepStatus = "pending"
    label: str
    message: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class LoopIteration(BaseModel):
    """One complete pass of the eval → diagnose → plan → train → verify cycle."""

    model_config = ConfigDict(extra="forbid")

    iteration_index: int = Field(ge=1)
    patch_strategy: str
    patch_cluster: str
    eval_before: float = Field(ge=0.0, le=1.0)
    eval_after: float | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed"] = "pending"


class AgentRunState(BaseModel):
    """Top-level state object for a SelfImprovementAgent run."""

    model_config = ConfigDict(extra="forbid")

    agent_run_id: str
    project_id: str
    target_kpi: float = Field(ge=0.0, le=1.0)
    max_iterations: int = Field(ge=1)
    diminishing_returns_threshold: float = Field(ge=0.0, le=1.0)
    current_iteration: int = Field(ge=0)
    status: AgentRunStatus = "idle"
    stop_reason: str | None = None
    iterations: list[LoopIteration] = Field(default_factory=list)
    reasoning_log: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AgentRunRequest(BaseModel):
    """Request to launch the self-improvement agent."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    checkpoint: str
    target_kpi: float = Field(default=0.75, ge=0.0, le=1.0)
    max_iterations: int = Field(default=3, ge=1, le=10)
    diminishing_returns_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    simulate: bool = True