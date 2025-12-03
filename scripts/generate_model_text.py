#!/usr/bin/env python3
"""
Generate text from the model at temperature 0 for use as training data.
This creates a dataset where ground truth = model predictions by definition.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def generate_text_batch(
    model,
    tokenizer,
    num_sequences: int,
    sequence_length: int,
    batch_size: int,
    device: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> str:
    """Generate text from model at low temperature for diversity while staying relatively deterministic."""

    model.eval()
    all_text = []

    print(f"Generating {num_sequences} sequences of {sequence_length} tokens each (temp={temperature})...")

    with torch.no_grad():
        for i in tqdm(range(0, num_sequences, batch_size), desc="Generating"):
            current_batch_size = min(batch_size, num_sequences - i)

            # Use varied English prompts to get diverse text
            prompts = [
                "The first thing to understand is that",
                "In the beginning",
                "The most important aspect of",
                "According to the research",
                "The study shows that",
                "The main reason for",
                "The history of",
                "The development of",
                "According to experts",
                "The data suggests that",
                "The analysis reveals",
                "The evidence indicates",
                "The results demonstrate",
                "The findings show",
                "The investigation uncovered",
                "The report concludes",
            ]

            # Use different prompt for each sequence in batch for more diversity
            batch_prompts = []
            for j in range(current_batch_size):
                prompt_idx = ((i + j) % len(prompts))
                batch_prompts.append(prompts[prompt_idx])

            # Tokenize with padding to handle different lengths
            encodings = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            batch_input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)

            # Generate with low temperature for diversity
            outputs = model.generate(
                batch_input_ids,
                max_new_tokens=sequence_length,
                do_sample=True,  # Sample for diversity
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=attention_mask,
            )

            # Decode and save
            for output in outputs:
                text = tokenizer.decode(output, skip_special_tokens=True)
                all_text.append(text)

    # Join all text with newlines
    combined_text = "\n\n".join(all_text)

    return combined_text


def main():
    parser = argparse.ArgumentParser(description="Generate text from model at temp=0")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m",
                        help="Model name to generate from")
    parser.add_argument("--num_sequences", type=int, default=2000,
                        help="Number of sequences to generate")
    parser.add_argument("--sequence_length", type=int, default=512,
                        help="Length of each generated sequence in tokens")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation")
    parser.add_argument("--output_file", type=str, default="generated_text.txt",
                        help="Output text file")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature during generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling probability mass")
    parser.add_argument("--repetition_penalty", type=float, default=1.05,
                        help="Penalty to discourage repeating the same tokens")
    parser.add_argument("--model_dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Determine dtype
    if args.model_dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    elif args.model_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.model_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading model {args.model_name}...")
    print(f"Using dtype: {dtype}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Ensure padding is defined for causal generation and mask creation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    # Generate text
    text = generate_text_batch(
        model,
        tokenizer,
        args.num_sequences,
        args.sequence_length,
        args.batch_size,
        args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # Save to file
    with open(args.output_file, 'w') as f:
        f.write(text)

    print(f"\nGenerated text saved to {args.output_file}")
    print(f"Total characters: {len(text)}")
    print(f"Estimated tokens: ~{len(text) // 4}")


if __name__ == "__main__":
    main()
