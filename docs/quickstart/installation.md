# Installation

Set up the full AlphaBrain runtime environment.

---

## System Requirements

- Python 3.10+
- CUDA 11.8+ / CUDA 12.x
- PyTorch 2.1+
- GPU ≥ 1 (A100 / H100 recommended)

---

## 1. Clone the Repository

```bash
git clone <repo-url> && cd VLA-Engine-Developer
```

---

## 2. Install Core Dependencies

```bash
conda create -n vla_engine_developer python=3.10 -y
conda activate vla_engine_developer
pip install -r requirements.txt
pip install -e .
```

Install Flash Attention:

```bash
pip install flash-attn --no-build-isolation
```

!!! tip "Flash Attention build time"
    The first `flash-attn` install compiles from source and takes 10–30 minutes.

---

## 3. Install Evaluation Dependencies

Evaluation uses a **separate conda environment** to avoid conflicts with the training environment. LIBERO example (see the [LIBERO official docs](https://libero-project.github.io/)):

```bash
conda create -n libero python=3.10 -y
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
pip install tyro matplotlib mediapy websockets msgpack rich "numpy==1.24.4"
```

---

## 4. Download Pretrained Models and Data

Place model weights under `PRETRAINED_MODELS_DIR` (default `data/pretrained_models/`), or use symlinks:

```
data/pretrained_models/
├── Qwen2.5-VL-3B-Instruct/
├── Qwen2.5-VL-7B-Instruct/
├── Qwen3-VL-4B-Instruct/
└── Qwen3-VL-9B-Instruct/
```

Download from HuggingFace:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir data/pretrained_models/Qwen2.5-VL-3B-Instruct
```

!!! tip "Model selection"
    | Model | Params | Recommended Use |
    |-------|--------|-----------------|
    | Qwen2.5-VL-3B-Instruct | 3B | Quick experiments, limited VRAM |
    | Qwen2.5-VL-7B-Instruct | 7B | High-accuracy tasks |
    | Qwen3-VL-4B-Instruct | 4B | Balanced performance and efficiency |
    | Qwen3-VL-9B-Instruct | 9B | Highest accuracy |

---

One-click LIBERO data download:

```bash
export DEST=/path/to/data/directory
bash examples/LIBERO/data_preparation.sh
```

The script downloads and symlinks the following datasets:

| Dataset | Description |
|---------|-------------|
| `libero_spatial` | Spatial understanding |
| `libero_object` | Object manipulation |
| `libero_goal` | Goal-directed |
| `libero_10` | Long-horizon |

Directory layout:

```
${LIBERO_DATA_ROOT}/
├── libero_spatial_no_noops_1.0.0_lerobot/
├── libero_object_no_noops_1.0.0_lerobot/
├── libero_goal_no_noops_1.0.0_lerobot/
└── libero_10_no_noops_1.0.0_lerobot/
```

---

## 5. Configure Environment Variables

Create `.env` at the project root:

```bash
# Data paths
LIBERO_DATA_ROOT=/your/path/to/IPEC-COMMUNITY
LIBERO_HOME=../LIBERO

# Python interpreter of the separate LIBERO evaluation conda env
LIBERO_PYTHON=/your/path/to/miniconda3/envs/libero/bin/python

# Pretrained model directory (relative to project root)
PRETRAINED_MODELS_DIR=data/pretrained_models
```

!!! note "`.env` is not version-controlled"
    `.env` is already in `.gitignore`, so you can safely store local paths and sensitive info.

---

## 6. Verify

```bash
python -c "import AlphaBrain; print('AlphaBrain ok')"
```

!!! warning "CUDA errors"
    Make sure your PyTorch build matches your CUDA version:
    ```bash
    python -c "import torch; print(torch.version.cuda)"
    ```

---

## Next

Head to [**Baseline VLA**](baselineVLA.md) for the default finetune + eval walk-through — a short run there also serves as an end-to-end check that your install works.
