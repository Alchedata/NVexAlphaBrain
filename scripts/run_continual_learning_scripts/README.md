# Continual Learning

One-command wrappers for AlphaBrain's **Continual Learning (CL)** pipeline:
train a single Vision-Language-Action (VLA) backbone sequentially over a
stream of manipulation tasks, with **Experience Replay (ER)** to mitigate
catastrophic forgetting, then evaluate the final checkpoint on the full
task matrix.

The pipeline is benchmark-agnostic: out of the box it covers the four
LIBERO suites; with a single flag it also runs on Robocasa365 or on any
user-defined stream of LeRobot-format task folders (see
[Custom task streams](#custom-task-streams-non-libero-benchmarks) below).

Four VLA architectures are supported, each with full-parameter and
**Low-Rank Adaptation (LoRA)** variants:

| Architecture  | Parameters | Backbone                               |
|:--------------|:-----------|:---------------------------------------|
| QwenGR00T     | ~3.8 B     | Qwen2.5-VL-3B + Flow-Matching DiT head |
| NeuroVLA      | ~3.0 B     | Qwen2.5-VL-3B + Q-Former + SNN head    |
| LlamaOFT      | ~11 B      | Llama-3.2-11B-Vision + MLP head        |
| PaliGemmaOFT  | ~3.0 B     | PaliGemma-3B + MLP head                |

---

## Results

We benchmark each architecture on **LIBERO-Goal**, training sequentially
on 10 tasks (50 demonstrations per task) and evaluating the final
checkpoint against the full 10-task matrix with 10 rollouts per task.
The reported **Average Success Rate (Avg SR)** is the mean over all
10 tasks; **Negative Backward Transfer (NBT)** measures how much
performance on earlier tasks drops as later tasks are learned (positive
values indicate forgetting is mitigated relative to naive sequential
fine-tuning).

<div align="center">

| Architecture  | Method                   | Avg SR   | NBT    |
|:--------------|:-------------------------|:--------:|:------:|
| QwenGR00T     | Full-parameter + ER      | ~45 %    | +0.15  |
| **QwenGR00T** | **LoRA (r=32) + ER**     | **~48 %**| **+0.15** |
| NeuroVLA      | Full-parameter + ER      | ~40 %    | +0.40  |
| NeuroVLA      | LoRA (r=32) + ER         | ~28 %    | +0.25  |
| LlamaOFT      | LoRA (r=16) + ER         | ~17 %    | +0.50  |

</div>

**Baseline** (naive sequential fine-tuning, no replay): **below 10 %**
across all architectures — catastrophic forgetting dominates.

> Numbers are conservative estimates from our internal runs; per-run
> variance is a few percentage points depending on seed, simulator
> state, attention implementation, and hardware. Reproduction results
> higher or lower than the table are expected and welcome via issues
> or pull requests.

---

## Quick start

All examples assume a fresh clone and `conda activate alphabrain`. Each
block `cd`s into this scripts directory so commands can be pasted
verbatim.

### Training

```bash
cd ./scripts/run_continual_learning_scripts

# Default — QwenGR00T LoRA + ER on LIBERO-Goal (~15 h on 2× A800)
bash run_cl_train.sh

# Smoke test — 5 steps × 10 tasks, ~3 min (pipeline check, not convergence)
bash run_cl_train.sh --smoke

# Switch architecture — NeuroVLA full-parameter + ER
bash run_cl_train.sh \
    --yaml configs/continual_learning/neurovla_continual_libero.yaml \
    --run-id neurovla_cl_run_v1

# Pin specific GPUs + custom step budget
bash run_cl_train.sh --gpus 1,2 -- \
    --continual_learning.steps_per_task=20000
```

Checkpoints are written to `results/Checkpoints/<run_id>/checkpoints/`:

| Variant           | Artifacts per task                                                          |
|:------------------|:----------------------------------------------------------------------------|
| LoRA              | `task_<k>_id<k>_steps_<N>_{lora_adapter/, action_model.pt, replay_state.json}` |
| Full-parameter    | `task_<k>_id<k>_steps_<N>_pytorch_model.pt`                                  |

### Evaluation

```bash
cd ./scripts/run_continual_learning_scripts

# Full 10×10 matrix — LoRA run (2 GPUs parallel)
bash run_cl_eval.sh \
    --run-id qwengr00t_cl_lora_libero_goal_v1 \
    --base-config configs/continual_learning/qwengr00t_cl_lora_libero.yaml \
    --gpus 0,1

# Full-parameter run (no --base-config needed)
bash run_cl_eval.sh \
    --run-id neurovla_cl_libero_goal_v1 --gpus 1

# Quick final-checkpoint sanity check (single GPU)
bash run_cl_eval.sh \
    --run-id qwengr00t_cl_lora_libero_goal_v1 \
    --base-config configs/continual_learning/qwengr00t_cl_lora_libero.yaml \
    --gpus 0 --last-only
```

Per-task success rates and the overall matrix are written to
`results/eval_cl/<run_id>/`.

---

## Custom task streams (non-LIBERO benchmarks)

The same `run_cl_train.sh` handles task streams beyond LIBERO. Users can
(a) select from ready-made Robocasa365 presets, or (b) define their own
stream inline in a YAML config.

```bash
cd ./scripts/run_continual_learning_scripts

# 1. One-time: point .env at the benchmark's LeRobot data root
echo "ROBOCASA365_DATA_DIR=/path/to/robocasa/v1.0" >> .env

# 2. Launch — 5 composite Robocasa365 tasks (QwenGR00T + LoRA + ER)
bash run_cl_train.sh \
    --yaml configs/continual_learning/qwengr00t_cl_lora_robocasa.yaml
```

**Defining a custom stream** — edit the yaml directly, no Python
changes required:

```yaml
continual_learning:
  task_sequence:
    base_data_mix: my_custom_mix        # must exist in DATASET_NAMED_MIXTURES
    num_tasks: 5
    task_order: [2, 0, 4, 1, 3]         # optional visit order
  task_stream_mode: by_dataset          # | by_task_index | auto
  steps_per_task: 5000
```

Partitioning modes:

| `task_stream_mode` | Semantics                                                            |
|:-------------------|:---------------------------------------------------------------------|
| `by_task_index`    | LIBERO default: partition one multi-task parquet by its `task_index` column. |
| `by_dataset`       | Robocasa-style: each sub-dataset in the mixture is one CL task.       |
| `auto`             | Try `by_task_index`; fall back to `by_dataset` if it yields < 2 tasks. |

Implementation notes, built-in Robocasa365 presets, and guidance for
adding new benchmarks are collected in
[`README_custom_streams.md`](README_custom_streams.md).

> **Evaluation scope.** `run_cl_eval.sh` currently launches the LIBERO
> simulator. Evaluation for Robocasa365 or other custom benchmarks
> requires wiring their respective simulation environment and is not
> yet covered by this wrapper.

---

## Prerequisites

```bash
conda activate alphabrain
cp .env.example .env
```

Edit `.env` with your local paths. Required:

| Variable                   | Purpose                                                                     |
|:---------------------------|:----------------------------------------------------------------------------|
| `PRETRAINED_MODELS_DIR`    | Parent directory holding `Qwen2.5-VL-3B-Instruct/`, `Llama-3.2-11B-Vision-Instruct/`, etc. |
| `LEROBOT_LIBERO_DATA_DIR`  | LeRobot-format LIBERO data root.                                            |
| `LIBERO_PYTHON`            | Python from a separate conda env containing `robosuite` and `libero` (eval-only). |
| `LIBERO_HOME`              | LIBERO project root (for simulator configuration paths).                    |

Optional (only for non-LIBERO streams):

| Variable                   | Purpose                                                                     |
|:---------------------------|:----------------------------------------------------------------------------|
| `ROBOCASA365_DATA_DIR`     | Root containing `target/composite/<TaskName>/<date>/lerobot/...`.           |

---

## CLI reference

### `run_cl_train.sh`

| Flag              | Description                                                                  | Default                                                  |
|:------------------|:-----------------------------------------------------------------------------|:---------------------------------------------------------|
| `--yaml PATH`     | CL config yaml (relative or absolute).                                       | `configs/continual_learning/qwengr00t_cl_lora_libero.yaml` |
| `--run-id ID`     | Override the yaml's `run_id` (controls the checkpoint directory name).       | from yaml                                                |
| `--gpus SPEC`     | Either a count (`"2"`) or a comma-separated id list (`"1,2,3"`). A list pins `CUDA_VISIBLE_DEVICES`. | auto-detect |
| `--port N`        | `accelerate` main process port.                                              | auto-select a free port                                  |
| `--smoke`         | 5 steps × all tasks × batch 4 — verifies the pipeline end-to-end.            | off                                                      |
| `--`              | Pass-through OmegaConf overrides (e.g. `--lora.rank=16`).                    | —                                                        |
| `-h`, `--help`    | Full help text.                                                              | —                                                        |

**Available yaml presets** (under `configs/continual_learning/`):

| Yaml                                                | Architecture   | Method                              |
|:----------------------------------------------------|:---------------|:------------------------------------|
| `qwengr00t_continual_libero.yaml`                   | QwenGR00T      | Full-parameter                      |
| **`qwengr00t_cl_lora_libero.yaml`** (default)       | QwenGR00T      | **LoRA (r=32)**                     |
| `qwengr00t_cl_lora_test.yaml`                       | QwenGR00T      | LoRA, smoke-sized                   |
| `qwengr00t_cl_lora_libero_spatial.yaml`             | QwenGR00T      | LoRA, LIBERO-Spatial                |
| `qwengr00t_cl_lora_robocasa.yaml`                   | QwenGR00T      | LoRA, Robocasa365 (5 composite tasks) |
| `neurovla_continual_libero.yaml`                    | NeuroVLA       | Full-parameter                      |
| `neurovla_cl_lora_libero.yaml`                      | NeuroVLA       | LoRA                                |
| `llama_oft_continual_libero.yaml`                   | LlamaOFT       | Frozen LLM                          |
| `llamaoft_cl_lora_libero.yaml`                      | LlamaOFT       | LoRA (r=16)                         |
| `paligemma_oft_continual_libero.yaml`               | PaliGemmaOFT   | Full-parameter                      |

### `run_cl_eval.sh`

| Flag                  | Description                                                                 | Default   |
|:----------------------|:----------------------------------------------------------------------------|:----------|
| `--run-id ID`         | **Required.** Run directory under `results/Checkpoints/`.                    | —         |
| `--base-config PATH`  | **Required for LoRA runs** — base yaml used to merge the adapter.           | —         |
| `--gpus LIST`         | Comma-separated GPU id list; determines parallelism.                        | `0`       |
| `--suite NAME`        | `libero_goal`, `libero_spatial`, `libero_object`, or `libero_10`.            | `libero_goal` |
| `--trials N`          | Rollouts per task.                                                          | `10`      |
| `--port-base N`       | Starting port (each parallel worker gets `+i`).                              | `5694`    |
| `--last-only`         | Evaluate only the final task checkpoint.                                    | off       |

The evaluator automatically:

1. Discovers every `task_*_lora_adapter/` or `task_*_pytorch_model.pt` under `<run_id>/checkpoints/`.
2. Detects LoRA runs and merges adapters into full checkpoints on demand (cached as `*_merged.pt`).
3. Parallelises across `--gpus` — each worker owns a dedicated policy server + port.
4. Emits per-checkpoint `eval.log` + `server.log` under `results/eval_cl/<run_id>/<checkpoint_name>/`.

---

## Architecture

```
scripts/run_continual_learning_scripts/run_cl_train.sh     (self-contained wrapper)
                                     │
                                     │  resolves --yaml, loads .env,
                                     │  probes framework + base VLM,
                                     │  exec accelerate launch
                                     ▼
AlphaBrain/training/continual_learning/train_custom.py     (entry)
                                     │
                                     │  activates custom-stream extensions
                                     │  (Robocasa365 presets, inline task_sequence,
                                     │   by-dataset partitioning), then delegates
                                     ▼
AlphaBrain/training/continual_learning/train.py            (trainer)
        │
        ├── algorithms/       ReplayBuffer + CLAlgorithm base
        ├── datasets/         TaskFilteredDataset + task_sequences
        └── trainer_utils/peft/
            apply_lora() · save_lora_checkpoint() · load_and_merge()
            merge_lora_checkpoint  (CLI for post-hoc adapter merging)
```

---

## Related components

| Component                                    | Path                                                                    |
|:---------------------------------------------|:------------------------------------------------------------------------|
| CL trainer                                   | `AlphaBrain/training/continual_learning/train.py`                       |
| Custom-stream extensions (add-only)          | `AlphaBrain/training/continual_learning/{train_custom,datasets/custom_streams}.py` |
| CL algorithms (ER; EWC / LwF / SI planned)   | `AlphaBrain/training/continual_learning/algorithms/`                    |
| Task sequences + `TaskFilteredDataset`       | `AlphaBrain/training/continual_learning/datasets/task_sequences.py`     |
| LoRA helpers (inject / save / load & merge)  | `AlphaBrain/training/trainer_utils/peft/`                               |
| YAML configurations                          | `configs/continual_learning/`                                           |
| Documentation hub                            | `docs/continual_learning/`                                              |
| mkdocs quickstart                            | `docs/quickstart/continual_learning.md`                                 |

---

## Tips and caveats

- **flash-attn ABI mismatch.** If the active environment has `torch ≥ 2.6`
  but `flash-attn` was built against `torch 2.2`, the default
  `attn_implementation: flash_attention_2` crashes at model load.
  Workaround — override to SDPA:
  ```bash
  bash run_cl_train.sh -- --framework.qwenvl.attn_implementation=sdpa
  ```
  or reinstall: `pip install flash-attn --no-build-isolation --force-reinstall`.
- **Eval uses a separate conda env.** `LIBERO_PYTHON` in `.env` must
  point to an interpreter with `robosuite` installed (distinct from the
  training env). The wrapper auto-detects and falls back to a
  conventional `vlacl_engine_eval` install when the configured path
  lacks the dependency.
- **LoRA evaluation caches merged checkpoints.** The first evaluation
  call merges each LoRA adapter into a `*_merged.pt` file (~7 GB per
  task). Subsequent calls reuse the cache; remove the file to force a
  re-merge.
- **Default run takes ~15 h on 2× A800 80 GB.** Use `--smoke` to verify
  the pipeline in three minutes before committing a full run.

---

## Further reading

- Full experiment record and 10×10 matrices: [`docs/continual_learning/EXPERIMENTS.md`](../../docs/continual_learning/EXPERIMENTS.md)
- Annotated experiment matrix: [`docs/continual_learning/EXPERIMENT_MATRIX.md`](../../docs/continual_learning/EXPERIMENT_MATRIX.md)
- Source-layout index: [`docs/continual_learning/CODE_LAYOUT.md`](../../docs/continual_learning/CODE_LAYOUT.md)
- Hosted quickstart (mkdocs): [`docs/quickstart/continual_learning.md`](../../docs/quickstart/continual_learning.md)
- Custom-stream implementation notes: [`README_custom_streams.md`](README_custom_streams.md)
