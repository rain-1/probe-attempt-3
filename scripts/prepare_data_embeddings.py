#!/usr/bin/env python3
"""
Embedding-based data preparation script.
Creates dataset mapping token embeddings -> binary labels (is this token the target?).
Much simpler than autoregressive setup - just direct classification on embeddings.
"""

import argparse
import json
import os
import random
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_token_id(tokenizer, token_str: str) -> int:
    """Get the token ID for a specific string."""
    tokens = tokenizer.encode(token_str, add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError(f"Token '{token_str}' encodes to {len(tokens)} tokens, expected 1")
    return tokens[0]


def extract_embedding_dataset(
    text: str,
    tokenizer,
    model,
    target_token_id: int,
    max_tokens: int = None,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract embeddings and labels from text.
    
    Returns:
        embeddings: (N, embed_dim) tensor of token embeddings
        labels: (N,) tensor of binary labels (1 if token == target, 0 otherwise)
    """
    print(f"Tokenizing text...")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    if max_tokens:
        token_ids = token_ids[:max_tokens]
    
    print(f"Total tokens: {len(token_ids)}")
    
    # Create labels (1 if token is target, 0 otherwise)
    labels = torch.tensor([1 if tid == target_token_id else 0 for tid in token_ids], dtype=torch.float32)
    
    print(f"Positive samples: {labels.sum().item()} / {len(labels)} ({100 * labels.mean().item():.2f}%)")
    
    # Get embeddings from the model's embedding layer
    print(f"Extracting embeddings...")
    embedding_layer = model.get_input_embeddings()
    
    # Process in batches to avoid memory issues
    batch_size = 1024
    all_embeddings = []
    
    for i in tqdm(range(0, len(token_ids), batch_size), desc="Extracting embeddings"):
        batch_token_ids = token_ids[i:i+batch_size]
        batch_tensor = torch.tensor([batch_token_ids], device=device)
        
        with torch.no_grad():
            batch_embeddings = embedding_layer(batch_tensor)  # (1, batch_len, embed_dim)
            all_embeddings.append(batch_embeddings[0].cpu())  # (batch_len, embed_dim)
    
    embeddings = torch.cat(all_embeddings, dim=0)  # (total_tokens, embed_dim)
    
    return embeddings, labels


def balance_and_split(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Balance positive/negative samples and split into train/val sets.
    Returns (train_embeddings, train_labels, val_embeddings, val_labels)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Get positive and negative indices
    pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
    
    print(f"Found {len(pos_indices)} positive and {len(neg_indices)} negative samples")
    
    # Balance: take min of both classes
    n_samples = min(len(pos_indices), len(neg_indices))
    print(f"Balancing to {n_samples} samples per class...")
    
    # Randomly sample to balance
    pos_sample_idx = torch.randperm(len(pos_indices))[:n_samples]
    neg_sample_idx = torch.randperm(len(neg_indices))[:n_samples]
    
    pos_indices = pos_indices[pos_sample_idx]
    neg_indices = neg_indices[neg_sample_idx]
    
    # Combine and shuffle
    all_indices = torch.cat([pos_indices, neg_indices])
    all_labels = torch.cat([torch.ones(n_samples), torch.zeros(n_samples)])
    
    # Shuffle
    shuffle_idx = torch.randperm(len(all_indices))
    all_indices = all_indices[shuffle_idx]
    all_labels = all_labels[shuffle_idx]
    
    # Get embeddings
    all_embeddings = embeddings[all_indices]
    
    # Split train/val
    split_idx = int(len(all_embeddings) * train_ratio)
    train_embeddings = all_embeddings[:split_idx]
    train_labels = all_labels[:split_idx]
    val_embeddings = all_embeddings[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"Train set: {len(train_embeddings)} samples ({train_labels.sum().item():.0f} positive)")
    print(f"Val set: {len(val_embeddings)} samples ({val_labels.sum().item():.0f} positive)")
    
    return train_embeddings, train_labels, val_embeddings, val_labels


def save_dataset(embeddings: torch.Tensor, labels: torch.Tensor, output_path: str, metadata: dict):
    """Save embeddings and labels to disk."""
    data = {
        "embeddings": embeddings,
        "labels": labels,
        "metadata": metadata,
    }
    torch.save(data, output_path)
    print(f"Saved {len(embeddings)} samples to {output_path}")


def get_model_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    if dtype_str == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    else:
        return torch.float32


def main():
    parser = argparse.ArgumentParser(description="Prepare embedding-based dataset")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input text file to process")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m",
                        help="Model name for tokenizer and embeddings")
    parser.add_argument("--target_token", type=str, default=" the",
                        help="Target token to predict")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum tokens to process (None = all)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of data to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data_embeddings",
                        help="Output directory for processed data")
    parser.add_argument("--model_dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model {args.model_name}...")
    dtype = get_model_dtype(args.model_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Get target token ID
    target_token_id = get_token_id(tokenizer, args.target_token)
    print(f"Target token '{args.target_token}' has ID: {target_token_id}")

    # Read text file
    print(f"Reading text from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        text = f.read()
    print(f"Loaded text with {len(text)} characters")

    # Extract embeddings and labels
    embeddings, labels = extract_embedding_dataset(
        text,
        tokenizer,
        model,
        target_token_id,
        max_tokens=args.max_tokens,
        device=args.device,
    )

    # Balance and split
    train_embeddings, train_labels, val_embeddings, val_labels = balance_and_split(
        embeddings,
        labels,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # Metadata
    metadata = {
        "model_name": args.model_name,
        "target_token": args.target_token,
        "target_token_id": target_token_id,
        "embedding_dim": embeddings.shape[1],
        "seed": args.seed,
        "data_type": "embeddings",
        "source_file": args.input_file,
    }

    # Save datasets
    train_path = os.path.join(args.output_dir, "train.pth")
    val_path = os.path.join(args.output_dir, "val.pth")

    save_dataset(train_embeddings, train_labels, train_path, metadata)
    save_dataset(val_embeddings, val_labels, val_path, metadata)

    # Save metadata separately for easy access
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\nDataset preparation complete!")
    print(f"Train samples: {len(train_embeddings)}")
    print(f"Val samples: {len(val_embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
