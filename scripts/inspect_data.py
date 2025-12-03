#!/usr/bin/env python3
"""
Data inspection utility to visualize and verify the prepared dataset.
"""

import argparse
import random

import torch
from transformers import AutoTokenizer


def inspect_dataset(data_path: str, model_name: str, num_samples: int = 5):
    """Load and inspect a dataset."""
    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path)

    chunks = data["chunks"]
    labels = data["labels"]
    metadata = data["metadata"]

    print("\n" + "=" * 80)
    print("DATASET METADATA")
    print("=" * 80)
    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"Total samples: {len(chunks)}")
    print(f"Positive samples: {sum(labels)} ({100 * sum(labels) / len(labels):.2f}%)")
    print(f"Negative samples: {len(labels) - sum(labels)} ({100 * (len(labels) - sum(labels)) / len(labels):.2f}%)")
    print(f"Context length: {len(chunks[0]) if chunks else 0}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Sample and display
    print("\n" + "=" * 80)
    print(f"RANDOM SAMPLES (showing {num_samples} positive and {num_samples} negative)")
    print("=" * 80)

    # Get positive and negative indices
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]

    # Sample
    sample_positive = random.sample(positive_indices, min(num_samples, len(positive_indices)))
    sample_negative = random.sample(negative_indices, min(num_samples, len(negative_indices)))

    print("\n--- POSITIVE SAMPLES (next token IS the target) ---\n")
    for idx in sample_positive:
        chunk = chunks[idx]
        text = tokenizer.decode(chunk)
        print(f"Sample {idx}:")
        #print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"  Text: {text}")
        print(f"  Label: {labels[idx]}")
        print()

    print("\n--- NEGATIVE SAMPLES (next token is NOT the target) ---\n")
    for idx in sample_negative:
        chunk = chunks[idx]
        text = tokenizer.decode(chunk)
        print(f"Sample {idx}:")
#        print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"  Text: {text}")
        print(f"  Label: {labels[idx]}")
        print()

    # Check for NaN or weird tokens
    print("\n" + "=" * 80)
    print("DATA VALIDATION")
    print("=" * 80)

    all_token_ids = set()
    for chunk in chunks:
        all_token_ids.update(chunk)

    print(f"Unique tokens in dataset: {len(all_token_ids)}")
    print(f"Token ID range: {min(all_token_ids)} to {max(all_token_ids)}")

    # Check vocab size
    vocab_size = tokenizer.vocab_size
    invalid_tokens = [tid for tid in all_token_ids if tid >= vocab_size]
    if invalid_tokens:
        print(f"WARNING: Found {len(invalid_tokens)} token IDs >= vocab_size ({vocab_size})")
    else:
        print(f"All token IDs are valid (< vocab_size={vocab_size})")

    print("\nDataset looks good!")


def main():
    parser = argparse.ArgumentParser(description="Inspect prepared dataset")
    parser.add_argument("--data_path", type=str, default="data/train.pth",
                        help="Path to dataset file")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m",
                        help="Model name for tokenizer")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to display per class")

    args = parser.parse_args()

    random.seed(42)
    inspect_dataset(args.data_path, args.model_name, args.num_samples)


if __name__ == "__main__":
    main()
