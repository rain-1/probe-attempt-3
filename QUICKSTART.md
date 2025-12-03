# Quick Start - Copy & Paste Commands

Get up and running quickly with copy-pasteable commands.

## Setup

```bash
# Install dependencies with uv
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt

# (Optional) Login to WandB for experiment tracking
wandb login
```

## Basic Workflow

### 1. Prepare Dataset

```bash
python scripts/prepare_data.py
```

### 2. Inspect Data (Verify Balance)

```bash
python scripts/inspect_data.py --data_path data/train.pth
python scripts/inspect_data.py --data_path data/val.pth
```

### 3. Train Probes (Layer Sweep)

```bash
python scripts/train_probe.py --layer_sweep auto --batch_size 48 --epochs 10
```

### 4. Visualize Results

```bash
python scripts/visualize_probe.py --num_tokens 4000 --output results.html
```

Then open `results.html` in your browser!

---

## Advanced Commands

### High-Quality Dataset (More Data)

```bash
python scripts/prepare_data.py \
  --max_texts 5000 \
  --max_chunks_per_class 100000 \
  --context_length 256
```

### Train Single Layer

```bash
python scripts/train_probe.py --layer 23 --batch_size 48 --epochs 15
```

### Train Specific Layers

```bash
python scripts/train_probe.py --layer_sweep 0,6,12,18,23 --batch_size 48
```

### Train with Custom WandB Group

```bash
python scripts/train_probe.py \
  --layer_sweep auto \
  --batch_size 48 \
  --wandb_project my-probes \
  --wandb_group experiment-1
```

### Current Token Prediction (Sanity Check)

```bash
python scripts/train_probe.py \
  --layer_sweep auto \
  --predict_current \
  --wandb_group sanity-check
```

### Visualize with Custom Text

```bash
echo "The quick brown fox jumps over the lazy dog." > test.txt

python scripts/visualize_probe.py \
  --text_source file \
  --text_file test.txt \
  --num_tokens 500 \
  --output custom_viz.html
```

### Train on Different Token

```bash
# Prepare data for " and" instead of " the"
python scripts/prepare_data.py --target_token " and"

# Train probes
python scripts/train_probe.py --layer_sweep auto
```

### Smaller Batch Size (Less VRAM)

```bash
python scripts/train_probe.py \
  --layer_sweep auto \
  --batch_size 24 \
  --epochs 10
```

---

## Full Pipeline Example

```bash
# Complete workflow from scratch
python scripts/prepare_data.py --max_texts 2000
python scripts/inspect_data.py
python scripts/train_probe.py --layer_sweep auto --batch_size 48 --epochs 12
python scripts/visualize_probe.py --output final_results.html
```

---

## Troubleshooting Quick Fixes

### CUDA Out of Memory
```bash
python scripts/train_probe.py --layer_sweep auto --batch_size 24
```

### Disable WandB
```bash
python scripts/train_probe.py --layer_sweep auto --no_wandb
```

### Use Different Model Dtype
```bash
python scripts/train_probe.py --model_dtype float32 --batch_size 32
```

---

## Expected Results

After running the basic workflow, you should see:

- **Data**: ~40k-100k balanced samples in `data/train.pth` and `data/val.pth`
- **Probes**: 5 probe files in `models/` (one per layer from auto sweep)
- **Performance**: Balanced accuracy between 70-80% for next token prediction
- **Visualization**: Interactive HTML showing prediction heatmap across layers

For detailed explanations, see [USAGE.md](USAGE.md).
