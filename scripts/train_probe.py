#!/usr/bin/env python3
"""
Probe training script with layer sweep support and WandB logging.
Trains linear probes to predict next token from hidden states.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")


class LinearProbe(nn.Module):
    """Simple linear probe for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def get_model_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    if dtype_str == "auto":
        # Use bfloat16 if available (CUDA), else float32
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str):
    """Load model and tokenizer."""
    print(f"Loading model {model_name} with dtype {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Model loaded on {device}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")

    return model, tokenizer


def extract_hidden_states(
    model,
    input_ids: torch.Tensor,
    layer: int,
    device: str,
    predict_current: bool = False,
) -> torch.Tensor:
    """
    Extract hidden states from a specific layer.
    Returns hidden states for the final token (or second-to-last if predict_current=True).
    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)

        if predict_current:
            # For current token prediction, use the last token's hidden state
            final_hidden = hidden_states[:, -1, :]
        else:
            # For next token prediction, use the last token's hidden state
            # (which predicts the next token)
            final_hidden = hidden_states[:, -1, :]

    return final_hidden


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute confusion matrix metrics."""
    preds_binary = (predictions >= 0.5).astype(int)

    tp = np.sum((preds_binary == 1) & (labels == 1))
    fp = np.sum((preds_binary == 1) & (labels == 0))
    tn = np.sum((preds_binary == 0) & (labels == 0))
    fn = np.sum((preds_binary == 0) & (labels == 1))

    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
    pos_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0
    neg_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (pos_accuracy + neg_accuracy) / 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = pos_accuracy
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "pos_accuracy": pos_accuracy,
        "neg_accuracy": neg_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def run_validation(
    probe: nn.Module,
    model,
    val_loader: DataLoader,
    layer: int,
    device: str,
    pos_weight_tensor: torch.Tensor,
    predict_current: bool,
    desc: str = "Validation",
) -> Tuple[float, Dict[str, float]]:
    """Run validation and return loss and metrics."""
    probe.eval()
    val_loss = 0.0
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch_input_ids, batch_labels in tqdm(val_loader, desc=desc, leave=False):
            batch_input_ids = batch_input_ids.to(device)
            batch_labels = batch_labels.to(device)

            hidden = extract_hidden_states(
                model,
                batch_input_ids,
                layer,
                device,
                predict_current=predict_current,
            )

            # Skip bad batches
            if torch.isnan(hidden).any() or torch.isinf(hidden).any():
                continue

            hidden = hidden.float()
            logits = probe(hidden).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch_labels,
                pos_weight=pos_weight_tensor,
            )

            val_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.extend(probs)
            val_true.extend(batch_labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    val_metrics = compute_metrics(np.array(val_preds), np.array(val_true))

    return avg_val_loss, val_metrics


def train_probe(
    model,
    tokenizer,
    train_data: dict,
    val_data: dict,
    layer: int,
    args,
    wandb_run=None,
) -> Tuple[LinearProbe, Dict[str, float]]:
    """Train a probe for a specific layer."""

    device = args.device
    hidden_size = model.config.hidden_size

    # Create probe (always in float32 for stability)
    probe = LinearProbe(hidden_size).to(device).float()

    # Prepare data
    train_chunks = torch.tensor(train_data["chunks"], dtype=torch.long)
    train_labels = torch.tensor(train_data["labels"], dtype=torch.float32)
    val_chunks = torch.tensor(val_data["chunks"], dtype=torch.long)
    val_labels = torch.tensor(val_data["labels"], dtype=torch.float32)

    train_dataset = TensorDataset(train_chunks, train_labels)
    val_dataset = TensorDataset(val_chunks, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create quick validation loader (subset)
    quick_val_size = max(1, int(len(val_dataset) * args.quick_val_fraction))
    quick_val_indices = list(range(quick_val_size))
    quick_val_subset = torch.utils.data.Subset(val_dataset, quick_val_indices)
    quick_val_loader = DataLoader(quick_val_subset, batch_size=args.batch_size, shuffle=False)

    # Calculate pos_weight for loss function
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight_raw = n_neg / n_pos if n_pos > 0 else 1.0
    # Apply sqrt and cap to avoid extreme reweighting
    pos_weight = min(math.sqrt(pos_weight_raw), args.pos_weight_cap)
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    print(f"Pos weight: {pos_weight:.3f} (raw: {pos_weight_raw:.3f}, capped at {args.pos_weight_cap})")

    # Optimizer
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    best_balanced_acc = 0.0
    best_metrics = None
    global_step = 0
    epoch_step = 0

    for epoch in range(args.epochs):
        probe.train()
        epoch_train_loss = 0.0
        epoch_train_preds = []
        epoch_train_true = []
        epoch_step = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_input_ids, batch_labels in pbar:
            # Check if we've hit steps_per_epoch limit
            if args.steps_per_epoch and epoch_step >= args.steps_per_epoch:
                break

            batch_input_ids = batch_input_ids.to(device)
            batch_labels = batch_labels.to(device)

            # Extract hidden states
            hidden = extract_hidden_states(
                model,
                batch_input_ids,
                layer,
                device,
                predict_current=args.predict_current,
            )

            # Check for NaN/Inf and skip bad batches
            if torch.isnan(hidden).any() or torch.isinf(hidden).any():
                print(f"Warning: NaN/Inf detected in hidden states, skipping batch")
                continue

            # Cast to float32 for probe
            hidden = hidden.float()

            # Forward pass
            logits = probe(hidden).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch_labels,
                pos_weight=pos_weight_tensor,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics for epoch summary
            epoch_train_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            epoch_train_preds.extend(probs)
            epoch_train_true.extend(batch_labels.cpu().numpy())

            # Compute batch metrics for this batch
            batch_preds = (probs >= 0.5).astype(int)
            batch_true = batch_labels.cpu().numpy()
            batch_acc = (batch_preds == batch_true).mean()

            # Format with fixed width to prevent jumping
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})

            # Log to wandb every step
            if wandb_run:
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                wandb_run.log({
                    "train/step_loss": loss.item(),
                    "train/step_accuracy": batch_acc,
                    "train/learning_rate": current_lr,
                    "global_step": global_step,
                })

            global_step += 1
            epoch_step += 1

            # Quick validation every N steps
            if args.val_every_n_steps and global_step % args.val_every_n_steps == 0:
                # Compute training metrics on accumulated data so far this epoch
                if len(epoch_train_preds) > 0:
                    quick_train_metrics = compute_metrics(
                        np.array(epoch_train_preds),
                        np.array(epoch_train_true)
                    )
                    quick_train_loss = epoch_train_loss / epoch_step
                else:
                    quick_train_metrics = None
                    quick_train_loss = 0.0

                quick_val_loss, quick_val_metrics = run_validation(
                    probe,
                    model,
                    quick_val_loader,
                    layer,
                    device,
                    pos_weight_tensor,
                    args.predict_current,
                    desc="Quick Val",
                )

                # Log quick validation and training metrics
                if wandb_run:
                    log_dict = {
                        "quick_val/loss": quick_val_loss,
                        "quick_val/balanced_accuracy": quick_val_metrics['balanced_accuracy'],
                        "quick_val/pos_accuracy": quick_val_metrics['pos_accuracy'],
                        "quick_val/neg_accuracy": quick_val_metrics['neg_accuracy'],
                        "global_step": global_step,
                    }

                    # Add training metrics if available
                    if quick_train_metrics:
                        log_dict.update({
                            "quick_train/loss": quick_train_loss,
                            "quick_train/balanced_accuracy": quick_train_metrics['balanced_accuracy'],
                            "quick_train/pos_accuracy": quick_train_metrics['pos_accuracy'],
                            "quick_train/neg_accuracy": quick_train_metrics['neg_accuracy'],
                        })

                    wandb_run.log(log_dict)

                print(f"\n[Step {global_step}] Quick Val - Loss: {quick_val_loss:.4f}, Bal Acc: {quick_val_metrics['balanced_accuracy']:.4f}")

                probe.train()  # Back to training mode

        # End-of-epoch full validation
        avg_val_loss, val_metrics = run_validation(
            probe,
            model,
            val_loader,
            layer,
            device,
            pos_weight_tensor,
            args.predict_current,
            desc="Full Validation",
        )

        # Compute training metrics
        train_metrics = compute_metrics(np.array(epoch_train_preds), np.array(epoch_train_true))
        avg_train_loss = epoch_train_loss / max(epoch_step, 1)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Balanced Acc: {train_metrics['balanced_accuracy']:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
        print(f"  Val Pos Acc: {val_metrics['pos_accuracy']:.4f}, Neg Acc: {val_metrics['neg_accuracy']:.4f}")

        # Log to wandb (end of epoch - full validation)
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "epoch_train/loss": avg_train_loss,
                "epoch_train/balanced_accuracy": train_metrics['balanced_accuracy'],
                "epoch_train/accuracy": train_metrics['accuracy'],
                "epoch_train/pos_accuracy": train_metrics['pos_accuracy'],
                "epoch_train/neg_accuracy": train_metrics['neg_accuracy'],
                "epoch_val/loss": avg_val_loss,
                "epoch_val/balanced_accuracy": val_metrics['balanced_accuracy'],
                "epoch_val/accuracy": val_metrics['accuracy'],
                "epoch_val/pos_accuracy": val_metrics['pos_accuracy'],
                "epoch_val/neg_accuracy": val_metrics['neg_accuracy'],
                "epoch_val/tp": val_metrics['tp'],
                "epoch_val/fp": val_metrics['fp'],
                "epoch_val/tn": val_metrics['tn'],
                "epoch_val/fn": val_metrics['fn'],
                "global_step": global_step,
            })

        # Track best model
        if val_metrics['balanced_accuracy'] > best_balanced_acc:
            best_balanced_acc = val_metrics['balanced_accuracy']
            best_metrics = val_metrics

    return probe, best_metrics


def save_probe(probe: LinearProbe, metadata: dict, output_path: str):
    """Save probe with embedded metadata."""
    save_dict = {
        "state_dict": probe.state_dict(),
        "metadata": metadata,
    }
    torch.save(save_dict, output_path)
    print(f"Saved probe to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train probe on hidden states")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m")
    parser.add_argument("--model_dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--train_data", type=str, default="data/train.pth")
    parser.add_argument("--val_data", type=str, default="data/val.pth")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to train probe on")
    parser.add_argument("--layer_sweep", type=str, default=None,
                        help="Comma-separated layer indices or 'auto' for automatic sweep")
    parser.add_argument("--predict_current", action="store_true",
                        help="Predict current token instead of next token")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="Limit steps per epoch (None = full epoch)")
    parser.add_argument("--val_every_n_steps", type=int, default=None,
                        help="Run validation every N steps (None = once per epoch)")
    parser.add_argument("--quick_val_fraction", type=float, default=0.1,
                        help="Fraction of val set to use for quick validation (default: 0.1)")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight_cap", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_project", type=str, default="probe-training")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once
    dtype = get_model_dtype(args.model_dtype)
    model, tokenizer = load_model_and_tokenizer(args.model_name, dtype, args.device)

    num_layers = model.config.num_hidden_layers

    # Determine layers to train on
    if args.layer is not None:
        layers = [args.layer]
    elif args.layer_sweep == "auto":
        # Auto sweep: first, last, and quartiles
        layers = [
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1,
        ]
        layers = sorted(set(layers))  # Remove duplicates
        print(f"Auto layer sweep: {layers}")
    elif args.layer_sweep:
        layers = [int(x.strip()) for x in args.layer_sweep.split(",")]
    else:
        # Default: last layer
        layers = [num_layers - 1]

    # Load data
    print(f"Loading training data from {args.train_data}...")
    train_data = torch.load(args.train_data, weights_only=False)
    print(f"Loading validation data from {args.val_data}...")
    val_data = torch.load(args.val_data, weights_only=False)

    # Initialize wandb group for sweeps
    if not args.no_wandb and WANDB_AVAILABLE and len(layers) > 1:
        if not args.wandb_group:
            task_type = "current" if args.predict_current else "next"
            args.wandb_group = f"layer-sweep-{task_type}"

    # Train probes for each layer
    results = []
    for layer_idx in layers:
        print(f"\n{'='*80}")
        print(f"Training probe for layer {layer_idx}/{num_layers-1}")
        print(f"{'='*80}\n")

        # Initialize wandb for this run
        wandb_run = None
        if not args.no_wandb and WANDB_AVAILABLE:
            task_type = "current" if args.predict_current else "next"
            run_name = f"probe-layer{layer_idx}-ctx{train_data['metadata']['context_length']}-{task_type}"

            wandb_run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=run_name,
                config={
                    "model_name": args.model_name,
                    "layer": layer_idx,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "predict_current": args.predict_current,
                    "context_length": train_data['metadata']['context_length'],
                    "target_token": train_data['metadata']['target_token'],
                },
                reinit=True,
            )

        # Train probe
        probe, best_metrics = train_probe(
            model,
            tokenizer,
            train_data,
            val_data,
            layer_idx,
            args,
            wandb_run,
        )

        # Save probe with metadata
        task_suffix = "current" if args.predict_current else "next"
        output_path = os.path.join(
            args.output_dir,
            f"probe_layer{layer_idx}_{task_suffix}.pth"
        )

        metadata = {
            "layer": layer_idx,
            "model_name": args.model_name,
            "task": task_suffix,
            "target_token": train_data['metadata']['target_token'],
            "target_token_id": train_data['metadata']['target_token_id'],
            "context_length": train_data['metadata']['context_length'],
            "hidden_size": model.config.hidden_size,
            "best_metrics": best_metrics,
        }

        save_probe(probe, metadata, output_path)

        results.append({
            "layer": layer_idx,
            "balanced_accuracy": best_metrics['balanced_accuracy'],
            "output_path": output_path,
        })

        if wandb_run:
            wandb_run.finish()

    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    for result in results:
        print(f"Layer {result['layer']}: Balanced Acc = {result['balanced_accuracy']:.4f}")
        print(f"  Saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
