# LESSONS LEARNED

Second attempt at training a probe, inspired by the Bengio probe paper. We chose a trivial goal: trying to figure out if the current inference pass is going to produce the token " the". If you attach the probe to the final layer this task should be trivial so you can expect success. That makes it a really good task to do for practice's sake.

## Numerical Stability (NaN/Inf issues)

- **float16 causes NaNs**: The gemma-3-270m model produces NaN values in hidden states when run in float16. This was a major blocker.
- **Solution**: Use bfloat16 (on CUDA) or float32. We added `--model_dtype auto` that picks bfloat16 when safely available.
- **Always check for NaN/Inf** in hidden states before training on them, and skip bad batches.
- **Cast probe to float32** even when model runs in bfloat16 - the probe's small linear layer benefits from full precision.

## Reward Hacking / Class Imbalance

- **Problem**: " the" is only ~4-5% of tokens. An unbalanced probe learns to always predict 0 (97% accuracy, 0% usefulness).
- **After rebalancing** with pos_weight, it reward-hacked the other way - always predicting 1.
- **Solution**: 
  1. Pre-balance the dataset (50/50 positive/negative chunks)
  2. Use **balanced accuracy** = (pos_acc + neg_acc) / 2 as the real metric
  3. Use sqrt-based pos_weight with a cap (e.g., max 5x) to dampen extreme reweighting
  4. Log the full confusion matrix (TP, FP, TN, FN) to WandB so you can see what's actually happening

## Metadata & Reproducibility

- **Bug**: Probe got attached to the wrong layer during visualization because the layer was passed as a CLI arg.
- **Solution**: Embed metadata in the probe.pth file itself:
  ```python
  {"state_dict": ..., "metadata": {"layer": 5, "task": "next", "model": "...", "target_token": " the"}}
  ```
- Visualization tools should **read from the probe file**, not rely on user to remember settings.

## Two Tasks: Current vs Next Token

- **Current token task** (`--predict_current`): "Is THIS token ' the'?" - Nearly trivial (~99%), just reading the embedding. Good for sanity checking but not interesting.
- **Next token task** (default): "Will the NEXT token be ' the'?" - Actually hard (~75% balanced acc). This tests if the model's hidden state encodes prediction information.

## Layer Sweep Insights

- Different layers have different predictive power
- Early layers (0-2): Surprisingly decent (~70-76% balanced acc) - some positional/contextual signal exists early
- Middle layers: Variable performance
- Later layers: Generally better for "next token" prediction as they're closer to the output
- Use `--layer_sweep auto` to test key layers: [0, Q1, mid, Q3, last]

## Performance & VRAM

- **Batch size matters**: 44-48 fits in 16GB VRAM for gemma-3-270m with context_length=256
- **Don't cache activations**: For large models, activation caching doesn't scale (TBs of storage). Just recompute.
- **DO cache tokenized chunks**: Small files, saves tokenization time across layer sweeps
- **Load model ONCE** for sweeps: Don't reload for each layer, pass preloaded model to train()

## Visualization

- **Interactive HTML** is great for exploring probe behavior
- **Layer slider** lets you see how predictions change through the network
- **Hover tooltips** showing exact probabilities help debugging
- **Text must wrap**: Don't use `&nbsp;` for spaces or `white-space: pre-wrap` - breaks line wrapping
- **NaN in JSON**: Python's `NaN` isn't valid JSON - convert to `null` when serializing

## WandB Best Practices

- Use **groups** for layer sweeps so runs are organized together
- Log **balanced_accuracy** as the primary metric (not raw accuracy)
- Log confusion matrix components (TP/FP/TN/FN) for post-hoc analysis
- Use descriptive run names: `probe-layer5-ctx256-all-next-gemma-3-270m`

## What's Next

- Try MLP probe instead of linear (more capacity)
- Try different target tokens (more predictable ones?)
- Try larger models where "next token" prediction might be better represented
- Investigate why some middle layers underperform

