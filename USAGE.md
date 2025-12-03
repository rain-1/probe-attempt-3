# Usage Guide

Complete guide to using the probe training system.

## Setup

### Install Dependencies

Using uv (recommended):
```bash
uv pip install -r requirements.txt
```

Or with pip:
```bash
pip install -r requirements.txt
```

### Configure WandB (Optional)

If you want to use WandB logging:
```bash
wandb login
```

Or set the API key in `.env`:
```bash
echo "WANDB_API_KEY=your_key_here" >> .env
```

## Workflow

### 1. Prepare Data

Download and process the Project Gutenberg dataset into balanced training/validation sets.

**Basic usage:**
```bash
python scripts/prepare_data.py
```

**Advanced options:**
```bash
python scripts/prepare_data.py \
  --model_name unsloth/gemma-3-270m \
  --target_token " the" \
  --context_length 256 \
  --max_texts 1000 \
  --max_chunks_per_class 50000 \
  --train_ratio 0.8 \
  --language en \
  --output_dir data
```

**Parameters:**
- `--model_name`: HuggingFace model name for tokenizer (default: `unsloth/gemma-3-270m`)
- `--target_token`: Token to predict (default: `" the"`)
- `--context_length`: Length of context window (default: `256`)
- `--max_texts`: Maximum texts to process from dataset (default: `1000`)
- `--max_chunks_per_class`: Max chunks before balancing (default: `50000`)
- `--train_ratio`: Train/val split ratio (default: `0.8`)
- `--language`: Language split to use - `en`, `fr`, `de`, `es`, `it`, `nl`, `pl`, `pt`, `ru`, `sv`, `zh` (default: `en`)
- `--output_dir`: Output directory (default: `data`)

**Output:**
- `data/train.pth`: Training dataset
- `data/val.pth`: Validation dataset
- `data/metadata.json`: Dataset metadata

---

### 2. Inspect Data

Verify the prepared dataset and view sample texts.

**Basic usage:**
```bash
python scripts/inspect_data.py
```

**Advanced options:**
```bash
python scripts/inspect_data.py \
  --data_path data/train.pth \
  --model_name unsloth/gemma-3-270m \
  --num_samples 5
```

**Parameters:**
- `--data_path`: Path to dataset file (default: `data/train.pth`)
- `--model_name`: Model name for tokenizer (default: `unsloth/gemma-3-270m`)
- `--num_samples`: Number of samples to display per class (default: `5`)

**What it shows:**
- Dataset statistics (total samples, class balance)
- Random positive/negative samples with decoded text
- Data validation (token ID checks, NaN detection)

---

### 3. Train Probes

Train linear probes on model hidden states.

**Single layer:**
```bash
python scripts/train_probe.py --layer 11
```

**Layer sweep (automatic):**
```bash
python scripts/train_probe.py --layer_sweep auto
```

**Layer sweep (manual):**
```bash
python scripts/train_probe.py --layer_sweep 0,5,11,17,23
```

**Full command with all options:**
```bash
python scripts/train_probe.py \
  --model_name unsloth/gemma-3-270m \
  --model_dtype auto \
  --train_data data/train.pth \
  --val_data data/val.pth \
  --output_dir models \
  --layer_sweep auto \
  --batch_size 48 \
  --epochs 10 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --pos_weight_cap 5.0 \
  --wandb_project probe-training \
  --wandb_group layer-sweep-next
```

**Parameters:**
- `--model_name`: Model to use (default: `unsloth/gemma-3-270m`)
- `--model_dtype`: Data type - `auto`, `bfloat16`, `float16`, `float32` (default: `auto`)
  - `auto`: Uses bfloat16 on CUDA if supported, else float32
  - **Important**: Use `bfloat16` or `float32`, NOT `float16` (causes NaN issues)
- `--train_data`: Path to training data (default: `data/train.pth`)
- `--val_data`: Path to validation data (default: `data/val.pth`)
- `--output_dir`: Directory to save probes (default: `models`)
- `--layer`: Single layer to train on
- `--layer_sweep`: Comma-separated layers or `auto` for automatic sweep
- `--predict_current`: Predict current token instead of next token
- `--batch_size`: Batch size (default: `48`, adjust based on VRAM)
- `--epochs`: Training epochs (default: `10`)
- `--learning_rate`: Learning rate (default: `1e-3`)
- `--weight_decay`: Weight decay for regularization (default: `1e-4`)
- `--pos_weight_cap`: Maximum pos_weight for class reweighting (default: `5.0`)
- `--wandb_project`: WandB project name (default: `probe-training`)
- `--wandb_group`: WandB group for organizing runs
- `--no_wandb`: Disable WandB logging

**Output:**
- `models/probe_layer{N}_next.pth`: Trained probe files with embedded metadata
- WandB logs (if enabled)

**Training Tips:**
- **Batch size**: 48 fits in 16GB VRAM for gemma-3-270m with context_length=256
- **Layer sweep**: Use `--layer_sweep auto` to test key layers: [0, Q1, mid, Q3, last]
- **Task types**:
  - Next token (default): "Will the NEXT token be ' the'?" - More interesting
  - Current token (`--predict_current`): "Is THIS token ' the'?" - Trivial (~99%)
- **Metrics**: Focus on `balanced_accuracy` (average of pos/neg accuracy), not raw accuracy

---

### 4. Visualize Results

Generate an interactive HTML visualization showing probe predictions across layers.

**Basic usage:**
```bash
python scripts/visualize_probe.py
```

**Advanced options:**
```bash
python scripts/visualize_probe.py \
  --probe_dir models \
  --model_name unsloth/gemma-3-270m \
  --model_dtype auto \
  --num_tokens 4000 \
  --text_source gutenberg \
  --output visualization.html
```

**Using custom text:**
```bash
python scripts/visualize_probe.py \
  --text_source file \
  --text_file my_text.txt \
  --num_tokens 2000 \
  --output my_visualization.html
```

**Parameters:**
- `--probe_dir`: Directory containing probe files (default: `models`)
- `--model_name`: Model name (uses probe metadata if not specified)
- `--model_dtype`: Model dtype (default: `auto`)
- `--num_tokens`: Number of tokens to visualize (default: `4000`)
- `--text_source`: Source of text - `gutenberg` or `file` (default: `gutenberg`)
- `--text_file`: Path to text file (required if `--text_source file`)
- `--language`: Language split for Gutenberg - `en`, `fr`, `de`, etc. (default: `en`)
- `--output`: Output HTML file path (default: `visualization.html`)

**Output:**
- Interactive HTML file with layer slider
- White background: Probe predicts 0 (NOT the target token)
- Red background: Probe predicts 1 (IS the target token)
- Hover over tokens to see exact probabilities

**Visualization Tips:**
- Use the layer slider to see how predictions change through the network
- Later layers typically show better "next token" prediction accuracy
- Hover tooltips show exact probability values for debugging

---

## Complete Example Workflow

```bash
# 1. Prepare data
python scripts/prepare_data.py \
  --max_texts 1000 \
  --context_length 256

# 2. Inspect the data
python scripts/inspect_data.py --data_path data/train.pth
python scripts/inspect_data.py --data_path data/val.pth

# 3. Train probes with layer sweep
python scripts/train_probe.py \
  --layer_sweep auto \
  --batch_size 48 \
  --epochs 10 \
  --wandb_group my-first-sweep

# 4. Generate visualization
python scripts/visualize_probe.py \
  --num_tokens 4000 \
  --output results.html

# 5. Open the visualization
# Open results.html in your web browser
```

---

## Interpreting Results

### Training Metrics

Key metrics logged to WandB and console:

- **balanced_accuracy**: (pos_acc + neg_acc) / 2 - PRIMARY METRIC
  - Focus on this, not raw accuracy
  - Accounts for class imbalance
  - Target: >70% is good, >80% is excellent for next token prediction

- **pos_accuracy**: True positive rate (recall)
  - How well does the probe catch actual ' the' tokens?

- **neg_accuracy**: True negative rate (specificity)
  - How well does the probe avoid false positives?

- **Confusion Matrix** (TP, FP, TN, FN):
  - Watch for reward hacking (always predicting 0 or 1)

### Layer Performance

Typical patterns:
- **Early layers (0-2)**: Decent performance (~70-76%) from positional/contextual signals
- **Middle layers**: Variable performance
- **Later layers**: Generally best for next token prediction

### Common Issues

**Low balanced accuracy (<60%)**:
- Model might not be encoding the prediction well at that layer
- Try a different layer or later in the network

**High pos_accuracy but low neg_accuracy (or vice versa)**:
- Reward hacking: probe learned to always predict one class
- Check `pos_weight_cap` parameter
- Verify dataset is balanced (use `inspect_data.py`)

**NaN/Inf errors**:
- NEVER use float16 with gemma-3-270m
- Use `--model_dtype auto` or `--model_dtype bfloat16`
- The script automatically skips bad batches

---

## File Organization

```
probe-attempt-3/
├── data/                    # Dataset files
│   ├── train.pth
│   ├── val.pth
│   └── metadata.json
├── models/                  # Trained probe files
│   ├── probe_layer0_next.pth
│   ├── probe_layer11_next.pth
│   └── ...
├── wandb_logs/             # WandB local logs
├── scripts/                # All executable scripts
│   ├── prepare_data.py
│   ├── inspect_data.py
│   ├── train_probe.py
│   └── visualize_probe.py
├── requirements.txt
├── DESIGN.md              # Project design doc
├── LESSONS.md             # Lessons learned
└── USAGE.md              # This file
```

---

## Tips & Best Practices

1. **Always inspect data first**: Use `inspect_data.py` to verify balance and sanity

2. **Start with layer sweep**: Use `--layer_sweep auto` to find the best layers

3. **Monitor WandB**: Track balanced_accuracy and confusion matrix in real-time

4. **Visualize results**: The HTML visualization reveals what the probe actually learned

5. **Adjust batch size**: If you get CUDA OOM errors, reduce `--batch_size`

6. **Use bfloat16**: Best balance of speed and numerical stability on CUDA

7. **Check for reward hacking**: If pos_accuracy or neg_accuracy is near 0% or 100%, the probe is cheating

---

## Advanced: Custom Tasks

### Predict a different token

```bash
# Prepare data for a different token
python scripts/prepare_data.py --target_token " and"

# Train probes
python scripts/train_probe.py --layer_sweep auto
```

### Predict current token (sanity check)

```bash
# This should get ~99% accuracy (trivial task)
python scripts/train_probe.py \
  --layer_sweep auto \
  --predict_current \
  --wandb_group sanity-check-current
```

### Smaller context window

```bash
# Use smaller context for faster training
python scripts/prepare_data.py --context_length 128

# Adjust batch size accordingly
python scripts/train_probe.py \
  --layer_sweep auto \
  --batch_size 64
```

---

## Troubleshooting

**Issue: `CUDA out of memory`**
- Reduce `--batch_size` (try 32, 24, or 16)
- Use `--model_dtype bfloat16` instead of float32

**Issue: `NaN in hidden states`**
- DO NOT use `--model_dtype float16`
- Use `--model_dtype auto` or `--model_dtype bfloat16`

**Issue: Probe always predicts 0 or 1**
- Dataset might not be balanced: run `inspect_data.py`
- Try adjusting `--pos_weight_cap` (lower it to 2.0 or 3.0)
- Check WandB confusion matrix

**Issue: WandB not logging**
- Run `wandb login` first
- Or use `--no_wandb` to disable
- Check that wandb is installed: `pip install wandb`

**Issue: Slow data preparation**
- Reduce `--max_texts` for faster iteration
- The script uses streaming, so it's already optimized

---

## Questions?

See [DESIGN.md](DESIGN.md) for project goals and [LESSONS.md](LESSONS.md) for insights from previous attempts.
