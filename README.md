# Probe Training System

Train linear probes to predict tokens from language model hidden states. A clean implementation incorporating lessons learned from previous attempts.

## Overview

This project trains interpretability probes on the [gemma-3-270m](https://huggingface.co/unsloth/gemma-3-270m) model using the [Project Gutenberg dataset](https://huggingface.co/datasets/manu/project_gutenberg). The goal is to predict whether the next token will be " the" based on hidden states from different layers.

**Key Features:**
- Balanced dataset preparation with 50/50 positive/negative samples
- Layer sweep support to test probes at different depths
- WandB integration for experiment tracking
- Interactive HTML visualization of predictions
- Numerical stability improvements (bfloat16, NaN handling)
- Embedded metadata in probe files for reproducibility

## Quick Start

```bash
# Install dependencies (using uv)
uv pip install -r requirements.txt

# 1. Prepare balanced dataset
python scripts/prepare_data.py

# 2. Inspect the data (optional but recommended)
python scripts/inspect_data.py

# 3. Train probes across multiple layers
python scripts/train_probe.py --layer_sweep auto --batch_size 48

# 4. Visualize results
python scripts/visualize_probe.py --output results.html
```

Then open `results.html` in your browser to see the interactive visualization!

## Project Structure

```
probe-attempt-3/
├── scripts/
│   ├── prepare_data.py      # Download and balance dataset
│   ├── inspect_data.py      # Verify and visualize data
│   ├── train_probe.py       # Train probes with layer sweep
│   └── visualize_probe.py   # Generate interactive HTML
├── data/                    # Processed datasets (.pth files)
├── models/                  # Trained probes (.pth files)
├── wandb_logs/             # WandB logs (if enabled)
├── requirements.txt
├── USAGE.md               # Detailed usage guide
├── DESIGN.md              # Project design document
└── LESSONS.md             # Lessons learned from attempts 1 & 2
```

## Key Concepts

### Two Prediction Tasks

1. **Next token prediction** (default): Predicts if the NEXT token will be " the"
   - More interesting and challenging (~75% balanced accuracy)
   - Tests if hidden states encode future token information

2. **Current token prediction** (`--predict_current`): Predicts if the CURRENT token is " the"
   - Trivial task (~99% accuracy)
   - Good for sanity checking

### Metrics

Focus on **balanced accuracy** = (positive_accuracy + negative_accuracy) / 2

This is crucial because:
- Raw accuracy is misleading with class imbalance
- A probe that always predicts "not the" gets 95% raw accuracy but 0% usefulness
- Balanced accuracy properly measures performance on both classes

### Layer Sweep

Different layers encode different information:
- **Early layers (0-2)**: Positional and contextual signals (~70-76%)
- **Middle layers**: Variable performance
- **Later layers**: Better next-token prediction (closer to output)

Use `--layer_sweep auto` to automatically test key layers.

## Documentation

- **[USAGE.md](USAGE.md)**: Complete usage guide with examples
- **[DESIGN.md](DESIGN.md)**: Project goals and architecture
- **[LESSONS.md](LESSONS.md)**: Important insights from previous attempts

## Requirements

- Python 3.8+
- CUDA-capable GPU with 16GB+ VRAM (recommended)
- ~20GB disk space for model and data

## Key Lessons Applied

Based on [LESSONS.md](LESSONS.md):

✅ **Numerical Stability**: Use bfloat16 or float32 (NOT float16)
✅ **Balanced Dataset**: Pre-balance to 50/50 with sqrt-capped pos_weight
✅ **Metadata Embedded**: Probes store layer, model, task info internally
✅ **Right Metrics**: Focus on balanced_accuracy, log confusion matrix
✅ **Efficient Caching**: Cache tokenized chunks, not activations
✅ **Interactive Viz**: HTML with layer slider and hover tooltips

## Example Results

After training with `--layer_sweep auto`:

```
Layer 0:  Balanced Acc = 0.7245
Layer 6:  Balanced Acc = 0.7412
Layer 11: Balanced Acc = 0.7598
Layer 17: Balanced Acc = 0.7689
Layer 23: Balanced Acc = 0.7523
```

The visualization shows how probe confidence varies across layers and helps identify what patterns each layer captures.

## Citation

Inspired by probing methods from:
- Alain & Bengio (2016) - Understanding intermediate layers using linear classifier probes

## License

MIT

## Contributing

This is attempt #3. See [LESSONS.md](LESSONS.md) for what we learned from attempts #1 and #2.
