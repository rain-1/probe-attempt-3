#!/usr/bin/env python3
"""
Data preparation script that reads from a single text file.
Useful for processing model-generated text where ground truth = model predictions.
"""

import argparse
import json
import os
import random
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def get_token_id(tokenizer, token_str: str) -> int:
    """Get the token ID for a specific string."""
    tokens = tokenizer.encode(token_str, add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError(f"Token '{token_str}' encodes to {len(tokens)} tokens, expected 1")
    return tokens[0]


def extract_chunks_from_text(
    text: str,
    tokenizer,
    target_token_id: int,
    context_length: int,
    task: str = "next",
    stride: int = None,
    max_chunks: int = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Extract chunks from a single text file.
    Returns (positive_chunks, negative_chunks).
    """
    positive_chunks = []
    negative_chunks = []

    print(f"Tokenizing text...")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens: {len(token_ids)}")

    if stride is None or stride <= 0:
        stride = context_length // 2
    print(f"Extracting chunks with stride={stride}...")

    for i in tqdm(range(0, len(token_ids) - context_length, stride), desc="Processing"):
        chunk = token_ids[i:i + context_length]

        if task == "next":
            if i + context_length >= len(token_ids):
                break
            target_token = token_ids[i + context_length]
        else:  # current token classification
            target_token = chunk[-1]

        if target_token == target_token_id:
            positive_chunks.append(chunk)
        else:
            negative_chunks.append(chunk)

        # Early exit if we have enough chunks
        if max_chunks and len(positive_chunks) > max_chunks and len(negative_chunks) > max_chunks:
            break

    print(f"Found {len(positive_chunks)} positive chunks and {len(negative_chunks)} negative chunks")
    return positive_chunks, negative_chunks


def balance_and_split(
    positive_chunks: List[List[int]],
    negative_chunks: List[int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[List[int]], List[int], List[List[int]], List[int]]:
    """
    Balance positive/negative samples and split into train/val sets.
    Returns (train_chunks, train_labels, val_chunks, val_labels)
    """
    random.seed(seed)

    # Balance: take min of both classes
    n_samples = min(len(positive_chunks), len(negative_chunks))
    print(f"Balancing to {n_samples} samples per class...")

    # Randomly sample to balance
    positive_chunks = random.sample(positive_chunks, n_samples)
    negative_chunks = random.sample(negative_chunks, n_samples)

    # Create labels
    positive_labels = [1] * len(positive_chunks)
    negative_labels = [0] * len(negative_chunks)

    # Combine and shuffle
    all_chunks = positive_chunks + negative_chunks
    all_labels = positive_labels + negative_labels

    combined = list(zip(all_chunks, all_labels))
    random.shuffle(combined)
    all_chunks, all_labels = zip(*combined)
    all_chunks = list(all_chunks)
    all_labels = list(all_labels)

    # Split train/val
    split_idx = int(len(all_chunks) * train_ratio)
    train_chunks = all_chunks[:split_idx]
    train_labels = all_labels[:split_idx]
    val_chunks = all_chunks[split_idx:]
    val_labels = all_labels[split_idx:]

    print(f"Train set: {len(train_chunks)} samples ({sum(train_labels)} positive)")
    print(f"Val set: {len(val_chunks)} samples ({sum(val_labels)} positive)")

    return train_chunks, train_labels, val_chunks, val_labels


def save_dataset(chunks: List[List[int]], labels: List[int], output_path: str, metadata: dict):
    """Save tokenized chunks and labels to disk."""
    data = {
        "chunks": chunks,
        "labels": labels,
        "metadata": metadata,
    }
    torch.save(data, output_path)
    print(f"Saved {len(chunks)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from text file")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input text file to process")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m",
                        help="Model name for tokenizer")
    parser.add_argument("--target_token", type=str, default=" the",
                        help="Target token to predict")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Length of context window")
    parser.add_argument("--stride", type=int, default=None,
                        help="Token stride between samples (default: half context for next-token, 1 for current-token)")
    parser.add_argument("--task", type=str, choices=["next", "current"], default="next",
                        help="Whether to label based on next token (default) or current token")
    parser.add_argument("--max_chunks_per_class", type=int, default=50000,
                        help="Maximum chunks to extract per class before balancing")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of data to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data_model_pred",
                        help="Output directory for processed data")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Get target token ID
    target_token_id = get_token_id(tokenizer, args.target_token)
    print(f"Target token '{args.target_token}' has ID: {target_token_id}")

    # Read text file
    print(f"Reading text from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        text = f.read()
    print(f"Loaded text with {len(text)} characters")

    # Extract chunks
    # Default stride: 1 for current-token so every position is eligible, else 50% overlap
    stride = args.stride if args.stride is not None else (1 if args.task == "current" else args.context_length // 2)

    positive_chunks, negative_chunks = extract_chunks_from_text(
        text,
        tokenizer,
        target_token_id,
        args.context_length,
        task=args.task,
        stride=stride,
        max_chunks=args.max_chunks_per_class,
    )

    # Balance and split
    train_chunks, train_labels, val_chunks, val_labels = balance_and_split(
        positive_chunks,
        negative_chunks,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # Metadata - mark as model_generated since this is model-generated text
    metadata = {
        "model_name": args.model_name,
        "target_token": args.target_token,
        "target_token_id": target_token_id,
        "context_length": args.context_length,
        "seed": args.seed,
        "label_source": "model_generated",  # Ground truth = model predictions
        "source_file": args.input_file,
        "task": args.task,
    }

    # Save datasets
    train_path = os.path.join(args.output_dir, "train.pth")
    val_path = os.path.join(args.output_dir, "val.pth")

    save_dataset(train_chunks, train_labels, train_path, metadata)
    save_dataset(val_chunks, val_labels, val_path, metadata)

    # Save metadata separately for easy access
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\nDataset preparation complete!")
    print(f"Train samples: {len(train_chunks)}")
    print(f"Val samples: {len(val_chunks)}")
    print(f"Balance: {sum(train_labels)}/{len(train_labels)} positive in train")


if __name__ == "__main__":
    main()
