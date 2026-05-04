# Nvex × AlphaBrain — Implementation Plan

**Last Updated:** April 25, 2026
**Based on:** PRD, existing codebase state, AlphaBrain capability audit

---

## Current State

### What exists and works today

| Component | Status | Notes |
|-----------|--------|-------|
| AlphaBrain VLA frameworks (OFT, GR00T, PI, NeuroVLA, CosmosPolicy, etc.) | ✅ Complete | 11 architectures, unified trainer |
| Training stack (Accelerate + DeepSpeed ZeRO-2) | ✅ Complete | Multi-GPU, W&B logging |
| Continual Learning module | ✅ Complete | LoRA-based, experience replay, cross-arch |
| RL fine-tuning (RLActionToken / TD3) | ✅ Complete | Requires 6 GPUs |
| World Model integration (Cosmos, Wan, V-JEPA) | ✅ Complete | 4 backbones |
| Benchmark suites (LIBERO, LIBERO-plus, Robocasa, Robocasa365) | ✅ Complete | Eval scripts + artifacts |
| Config system (YAML + modes + CLI overrides) | ✅ Complete | |
| Deployment module (model_server, upload) | ✅ Partial | Basic server exists, not productionized |
| Nvex investor demo HTML (`demo/nvex-demo.html`) | ✅ Complete | All 7 pages, fully interactive |
| React demo app (`demo/src/`) | ✅ Complete | All 7 pages implemented with shared components, mock data, and build validation |
| Nvex backend / orchestration logic | ✅ Complete | `nvex_server/` now provides export, planning, dispatch, polling, report, and demo bootstrap endpoints |
| Real AlphaBrain ↔ Nvex job interface | ✅ Partial | `JobDispatcher` wraps AlphaBrain shell entry points and supports file-backed polling plus simulated demo jobs |

---

## Milestone 1 — Narrative MVP (Demo-Ready) ✅ Complete

**Goal:** A polished, clickable demo that tells the full Nvex story end-to-end. All data can be mocked or pre-generated.

### Deliverables
- [x] `demo/nvex-demo.html` — standalone 7-page demo, all pages implemented
- [x] README.md rewritten to position Nvex as orchestration layer for both investors and customers
- [x] PRD finalized (`prd.md`)
- [x] Frontend wireframe/IA documented (`frontend-design.md`)
- [x] React demo app — build all 7 pages and components to match `nvex-demo.html`
- [x] Add a second demo scenario (non-LIBERO) to show breadth

### React App Components Needed
```
demo/src/
├── components/
│   ├── Sidebar.jsx
│   ├── TopBar.jsx
│   ├── KPICard.jsx
│   ├── FailureCluster.jsx
│   ├── RadarChart.jsx
│   ├── TimelineStep.jsx
│   └── AssetCard.jsx
├── pages/
│   ├── Home.jsx
│   ├── ProjectOverview.jsx
│   ├── FailureMap.jsx
│   ├── PatchPlan.jsx
│   ├── IterationRunner.jsx
│   ├── ImprovementReport.jsx
│   └── PlatformMemory.jsx
└── data/
    └── mockData.js
```

### Milestone 1 Exit Criteria Reached
- React demo builds successfully with Vite (`npm run build`)
- Shared component layer now covers KPI cards, failure clusters, radar chart, timeline steps, and asset cards
- The demo includes breadth beyond LIBERO via an additional RoboCasa scenario on the hub
- The standalone HTML and React demo now tell the same investor narrative at the page level

---

## Milestone 2 — Executable MVP (Real Loop) ✅ Complete

**Goal:** Wire at least one real AlphaBrain execution path into the Nvex demo. Produce a genuine before/after improvement artifact.

### 2A — Real Eval Artifact Ingestion
- [x] Define `EvalRun` schema (see PRD §8.2)
- [x] Write an AlphaBrain eval artifact exporter: converts benchmark output to `EvalRun` JSON
- [x] Load real LIBERO eval results into the Failure Map page
- [x] Replace mocked failure clusters with real per-task breakdown

### 2B — Patch Plan Engine (Rule-Based v1)
- [x] Implement `PatchPlanGenerator` — maps failure cluster patterns to training strategy recommendations
  - Rule: occlusion failures → target diverse viewpoint data + CL update
  - Rule: recovery gaps → teleop corrections + fine-tune
  - Rule: language variation failures → language augmentation + VLM co-training
- [x] Output structured `PatchPlan` JSON (see PRD §8.3)
- [x] Connect Patch Plan page to live generator

### 2C — AlphaBrain Job Interface
- [x] Define `IterationJob` schema: `plan_id`, `execution_backend`, `checkpoint`, `config`
- [x] Implement `JobDispatcher`: wraps AlphaBrain training scripts as callable jobs
  - Support `alphabrain_cl` (continual learning)
  - Support `alphabrain_finetune` (baseline fine-tune)
  - Support `alphabrain_eval` (re-evaluation only)
- [x] Implement job status polling (file-based or lightweight queue)
- [x] Wire Iteration Runner page to live job status

### 2D — Improvement Report from Real Artifacts
- [x] Load before/after eval artifacts and compute actual uplift
- [x] Save patch recipe to Platform Memory as a `ReusableAsset`
- [x] Produce at least one real improvement case: LIBERO Kitchen, 62% → 74%

### Infrastructure for Milestone 2
- [x] FastAPI service (`nvex_server/`) wrapping the orchestration logic
- [x] `POST /api/eval/import` — ingest eval artifact
- [x] `POST /api/plan/generate` — run PatchPlanGenerator
- [x] `POST /api/iteration/start` — dispatch job to AlphaBrain
- [x] `GET /api/iteration/{id}/status` — poll job progress
- [x] `GET /api/report/{iteration_id}` — fetch improvement report
- [x] Update React demo to consume these endpoints

---

## Milestone 3 — Self-Improving Agent ✅ Complete

**Goal:** Nvex runs the full failure-to-fix loop autonomously, without human intervention at each step. See [`SELF_IMPROVEMENT_AGENT.md`](SELF_IMPROVEMENT_AGENT.md) for full design.

### 3A — Autonomous Loop Orchestrator
- [x] Implement `SelfImprovementAgent` — orchestrator that:
  - Triggers eval on a checkpoint
  - Reads the EvalRun and identifies failure clusters
  - Selects the highest-leverage patch strategy
  - Dispatches to AlphaBrain
  - Evaluates the result
  - Decides whether to iterate again or terminate
- [x] Add stopping criteria: target KPI reached, max iterations, diminishing returns threshold
- [x] Add structured logging of agent reasoning at each step

### 3B — Agent Tool Registry
- [x] `run_eval(checkpoint, benchmark)` → EvalRun
- [x] `diagnose_failures(eval_run)` → FailureDiagnosis
- [x] `generate_patch_plan(diagnosis)` → PatchPlan
- [x] `dispatch_training(plan)` → IterationJob
- [x] `compare_checkpoints(before, after)` → ImprovementReport
- [x] `save_to_memory(asset)` → ReusableAsset

### 3C — Demo Mode for Self-Improving Agent
- [x] Add "Auto-Improve" button on Iteration Runner page
- [x] Animate the full loop: each step highlights as the agent processes it
- [x] Show agent reasoning panel (why it chose CL over SFT, why it targeted occlusion data)
- [x] Show multi-iteration view: loop 1 (62→74%), loop 2 (74→81%), convergence at 85%

### 3D — LLM Integration
- [x] `LLMNarrator` — uses OpenAI (gpt-4o-mini) when `OPENAI_API_KEY` is set, falls back to deterministic templates
- [x] Natural-language narration for diagnosis, plan, verify, and stop-check steps

### Infrastructure for Milestone 3
- [x] `nvex_server/agent.py` — `SelfImprovementAgent` with demo (precomputed replay) and real modes
- [x] `nvex_server/llm_narrator.py` — LLM narration with OpenAI + template fallback
- [x] New schemas: `FailureDiagnosis`, `AgentStep`, `LoopIteration`, `AgentRunState`, `AgentRunRequest`
- [x] `POST /api/agent/run` — launch autonomous loop
- [x] `GET /api/agent/{id}/status` — poll agent state
- [x] `POST /api/agent/{id}/advance` — advance one step (demo mode animation)
- [x] `GET /api/demo/agent` — pre-seeded demo agent run
- [x] `demo/src/components/AgentReasoningPanel.jsx` — step-by-step reasoning UI
- [x] `demo/src/components/MultiIterationChart.jsx` — pure-SVG multi-loop chart
- [x] Updated `IterationRunner.jsx` — Auto-Improve toggle, loop progress bar, reasoning panel
- [x] Updated `ImprovementReport.jsx` — multi-loop comparison chart, stop-reason callout

---

## Milestone 4 — Customer-Grade Platform

**Goal:** Extend from a single demo scenario to a multi-project, multi-user platform.

### P0 Tickets (Investor-Critical, Build Now)

- [ ] **M4-P0-01: Streaming Agent Timeline + Variable Step Durations**
  - Scope: backend emits ordered agent events; frontend shows live timeline and auto-play controls.
  - Acceptance criteria: run shows `run_started`/`step_started`/`step_completed`/`run_stopped` events in sequence; demo can play without manual clicking.
  - Status: **In Progress**

- [ ] **M4-P0-02: Multi-Iteration Arc with Non-Monotonic Reality**
  - Scope: demo arc includes at least one regression before recovery; chart and report show per-loop deltas.
  - Acceptance criteria: one loop has negative delta and is visibly rendered as regression.
  - Status: **In Progress**

- [ ] **M4-P0-03: Rollback Event + Recovery Loop**
  - Scope: stopping logic can emit rollback event, mark loop as rolled back, and continue from prior checkpoint baseline.
  - Acceptance criteria: timeline contains a rollback event and follow-up loop resumes from rollback baseline.
  - Status: **In Progress**

- [ ] **M4-P0-04: Multi-Project Isolation in Backend Store**
  - Scope: split global in-memory maps into project-scoped collections and enforce project_id on agent/eval/plan/iteration routes.
  - Acceptance criteria: project A data never appears in project B responses.

- [ ] **M4-P0-05: Persistent Platform Memory (File/DB-backed)**
  - Scope: replace volatile `InMemoryStore` memory assets with persistent repository (SQLite or file-backed JSON).
  - Acceptance criteria: server restart preserves recipes, templates, and failure patterns.

### P1 Tickets (Customer Readiness)

- [ ] **M4-P1-01: Customer Onboarding API (BYO Checkpoint + Eval Artifact)**
  - Scope: guided endpoints for registering a project, uploading checkpoint metadata, and importing benchmark artifacts.
  - Acceptance criteria: new customer project can be created and run through eval -> plan without code edits.

- [ ] **M4-P1-02: Benchmark Connector Expansion (RoboCasa/Tabletop/Custom)**
  - Scope: unify exporter adapters and normalize imported metrics across suites.
  - Acceptance criteria: at least 3 suites render correctly in Failure Map and Improvement Report.

- [ ] **M4-P1-03: Role-Based Views (Operator vs Executive)**
  - Scope: frontend route guards and dashboard tailoring by role.
  - Acceptance criteria: operator sees full execution logs; executive sees KPI/ROI view with guardrail summaries.

- [ ] **M4-P1-04: Governance + Audit Trail**
  - Scope: store full run decisions, tool inputs/outputs, rollback triggers, and approval checkpoints.
  - Acceptance criteria: each run has an exportable audit log bundle.

### P2 Tickets (Scale + Enterprise)

- [ ] **M4-P2-01: External Integration API**
  - Scope: webhooks/REST endpoints for external eval pipelines and training infra.
  - Acceptance criteria: external system can push eval results and receive patch plans.

- [ ] **M4-P2-02: Cost/ROI Observatory**
  - Scope: per-iteration compute/time/cost tracking with ROI rollups.
  - Acceptance criteria: report displays uplift per dollar and estimated monthly run cost.

- [ ] **M4-P2-03: Security Hardening + SOC2 Readiness Track**
  - Scope: authn/authz baseline, secrets handling, logging controls, dependency audit checklist.
  - Acceptance criteria: security checklist documented and first-pass audit completed.

---

## Priority Order for Next Sprint

| Priority | Task | Milestone | Effort |
|----------|------|-----------|--------|
| 🔴 High | M4-P0-01 Streaming timeline + auto-play controls | M4 | ~1.5 days |
| 🔴 High | M4-P0-02 Multi-iteration non-monotonic arc (regression) | M4 | ~1 day |
| 🔴 High | M4-P0-03 Rollback event + recovery loop | M4 | ~1 day |
| 🔴 High | M4-P0-04 Multi-project isolation layer | M4 | ~2 days |
| 🟡 Med | M4-P0-05 Persistent platform memory backend | M4 | ~2 days |
| 🟡 Med | M4-P1-01 Customer onboarding flow (BYO artifacts/checkpoint) | M4 | ~2 days |
| 🟢 Low | M4-P1-03 Role-based operator/executive views | M4 | ~2 days |
