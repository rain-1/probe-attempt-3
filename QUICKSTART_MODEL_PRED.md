# Quick Start: Model Prediction Dataset

This guide shows how to create a dataset where labels are based on model predictions rather than ground truth.

## Approach

Instead of labeling based on what token *actually* appears next in text, we label based on what the model *predicts* will appear next. This tests if the probe can predict the model's own behavior.

## Two Methods

### Method 1: Generate Model Text (Recommended - Fast)

Generate text from the model at temperature 0, then process it normally:

```bash
# Step 1: Generate text from model (batched, efficient)
python scripts/generate_model_text.py \
  --num_sequences 2000 \
  --sequence_length 512 \
  --batch_size 16 \
  --output_file generated_text.txt

# Step 2: Process the generated text
python scripts/prepare_data_from_file.py \
  --input_file generated_text.txt \
  --output_dir data_model_pred

# Step 3: Train probe
python scripts/train_probe.py \
  --train_data data_model_pred/train.pth \
  --val_data data_model_pred/val.pth \
  --layer 0 \
  --epochs 1 \
  --steps_per_epoch 200 \
  --batch_size 32 \
  --no_wandb
```

**Why this is fast:**
- Model runs once per sequence (not once per chunk)
- Batched generation is very efficient
- Ground truth = model predictions by definition (since model generated the text)
- ETA: ~10 minutes for full dataset generation

### Method 2: Per-Chunk Prediction (Slow - Not Recommended)

Run model inference on every chunk from real text:

```bash
python scripts/prepare_data.py \
  --use_model_predictions \
  --output_dir data_model_pred
```

**Why this is slow:**
- Model runs once per chunk (~100k times)
- No batching possible
- ETA: Several hours

## What You're Measuring

**Ground truth labels:**
- "Does this hidden state encode information about what token *actually* comes next?"
- Tests if the model internally represents linguistic patterns

**Model prediction labels:**
- "Does this hidden state encode information about what the model *will predict*?"
- Tests if the probe can replicate the model's decision-making

## Expected Results

Probes trained on model predictions should achieve **higher accuracy** because:
- The task is easier (predict model behavior vs. predict reality)
- The information is definitely present in the hidden states (it's what the model uses)
- Less noise from model errors

## Comparison

Train probes on both datasets and compare:

```bash
# Ground truth (existing data)
python scripts/train_probe.py --train_data data/train.pth --val_data data/val.pth --layer 0 --no_wandb

# Model predictions
python scripts/train_probe.py --train_data data_model_pred/train.pth --val_data data_model_pred/val.pth --layer 0 --no_wandb
```

The difference in accuracy tells you how much "knowable but unused" information is in the hidden states.
