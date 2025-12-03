#!/usr/bin/env python3
"""
Visualize embedding-based probe predictions.
Shows probe predictions on token embeddings with a simple interface.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class LinearProbe(nn.Module):
    """Simple linear probe for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def load_probe(probe_path: str, device: str):
    """Load probe from file with metadata."""
    data = torch.load(probe_path, map_location=device, weights_only=False)
    metadata = data["metadata"]

    probe = LinearProbe(metadata["embedding_dim"]).to(device)
    probe.load_state_dict(data["state_dict"])
    probe.eval()

    return probe, metadata


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


def generate_predictions(
    model,
    probe: nn.Module,
    token_ids: List[int],
    device: str,
) -> List[float]:
    """
    Generate predictions for each token embedding.
    Returns list of probabilities (one per token).
    """
    # Get embedding layer
    embedding_layer = model.get_input_embeddings()
    
    # Process in batches
    batch_size = 1024
    all_predictions = []
    
    for i in tqdm(range(0, len(token_ids), batch_size), desc="Generating predictions"):
        batch_token_ids = token_ids[i:i+batch_size]
        batch_tensor = torch.tensor([batch_token_ids], device=device)
        
        with torch.no_grad():
            # Get embeddings
            batch_embeddings = embedding_layer(batch_tensor)  # (1, batch_len, embed_dim)
            batch_embeddings = batch_embeddings[0].float()  # (batch_len, embed_dim)
            
            # Get predictions
            logits = probe(batch_embeddings).squeeze(-1)  # (batch_len,)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_predictions.extend(probs.tolist())
    
    return all_predictions


def create_html_visualization(
    text: str,
    tokens: List[str],
    predictions: List[float],
    metadata: dict,
    output_path: str,
):
    """Create interactive HTML visualization."""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Probe Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        h1 {{
            color: #333;
            text-align: center;
        }}

        .metadata {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .metadata p {{
            margin: 5px 0;
            color: #666;
        }}

        .text-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            line-height: 1.8;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}

        .token {{
            display: inline;
            padding: 2px 0;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }}

        .token:hover {{
            filter: brightness(0.9);
        }}

        .legend {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .legend-gradient {{
            height: 30px;
            background: linear-gradient(to right,
                rgba(255, 255, 255, 1) 0%,
                rgba(255, 0, 0, 0.3) 50%,
                rgba(255, 0, 0, 1) 100%);
            border-radius: 4px;
            margin: 10px 0;
        }}

        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #666;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            white-space: nowrap;
        }}

        .stats {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .stats h3 {{
            margin-top: 0;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}

        .stat-item {{
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}

        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>Embedding Probe Visualization</h1>

    <div class="metadata">
        <p><strong>Model:</strong> {metadata.get('model_name', 'N/A')}</p>
        <p><strong>Target Token:</strong> "{metadata.get('target_token', 'N/A')}"</p>
        <p><strong>Type:</strong> Embedding-based (direct token classification)</p>
        <p><strong>Embedding Dimension:</strong> {metadata.get('embedding_dim', 'N/A')}</p>
    </div>

    <div class="text-container" id="textContainer">
        <!-- Text will be inserted here -->
    </div>

    <div class="legend">
        <h3>Prediction Probability</h3>
        <div class="legend-gradient"></div>
        <div class="legend-labels">
            <span>0.0 (Definitely NOT)</span>
            <span>0.5 (Uncertain)</span>
            <span>1.0 (Definitely YES)</span>
        </div>
        <p style="font-size:14px;color:#666;margin-top:10px;">
            Colors indicate the probe's belief that each token is "{metadata.get('target_token', 'N/A').strip()}".
        </p>
    </div>

    <div class="stats" id="stats">
        <h3>Statistics</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value" id="totalTokens">-</div>
                <div class="stat-label">Total Tokens</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="avgProb">-</div>
                <div class="stat-label">Avg Probability</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="highConfidence">-</div>
                <div class="stat-label">High Confidence (>0.8)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="actualTarget">-</div>
                <div class="stat-label">Actual Target Tokens</div>
            </div>
        </div>
    </div>

    <script>
        const tokens = {json.dumps(tokens)};
        const predictions = {json.dumps(predictions)};
        const targetToken = {json.dumps(metadata.get('target_token', ''))};

        const textContainer = document.getElementById('textContainer');
        let tooltip = null;

        function getColorForProbability(prob) {{
            if (prob === null || prob === undefined || isNaN(prob)) {{
                return 'rgba(128, 128, 128, 0.2)';
            }}

            const alpha = Math.abs(prob - 0.5) * 2;
            const red = Math.round(255 * prob);
            const green = Math.round(255 * (1 - prob));
            const blue = Math.round(255 * (1 - prob));

            return `rgba(${{red}}, ${{green}}, ${{blue}}, ${{alpha * 0.6 + 0.1}})`;
        }}

        function renderText() {{
            let html = '';
            for (let i = 0; i < tokens.length; i++) {{
                const token = tokens[i];
                const prob = predictions[i];
                const color = getColorForProbability(prob);
                const probText = (prob !== null && prob !== undefined && !isNaN(prob))
                    ? prob.toFixed(3)
                    : 'N/A';

                html += `<span class="token" style="background-color: ${{color}};" data-prob="${{probText}}" data-token="${{i}}">${{token}}</span>`;
            }}

            textContainer.innerHTML = html;

            // Add hover listeners
            document.querySelectorAll('.token').forEach(tokenEl => {{
                tokenEl.addEventListener('mouseenter', showTooltip);
                tokenEl.addEventListener('mouseleave', hideTooltip);
            }});

            // Compute statistics
            const validPreds = predictions.filter(p => p !== null && !isNaN(p));
            const avgProb = validPreds.reduce((a, b) => a + b, 0) / validPreds.length;
            const highConfidence = validPreds.filter(p => p > 0.8).length;
            const actualTarget = tokens.filter(t => t === targetToken).length;

            document.getElementById('totalTokens').textContent = tokens.length;
            document.getElementById('avgProb').textContent = avgProb.toFixed(3);
            document.getElementById('highConfidence').textContent = highConfidence;
            document.getElementById('actualTarget').textContent = actualTarget;
        }}

        function showTooltip(event) {{
            const prob = event.target.getAttribute('data-prob');
            const tokenIndex = event.target.getAttribute('data-token');

            if (!tooltip) {{
                tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                document.body.appendChild(tooltip);
            }}

            tooltip.textContent = `Token ${{tokenIndex}}: P = ${{prob}}`;
            tooltip.style.display = 'block';

            const rect = event.target.getBoundingClientRect();
            tooltip.style.left = rect.left + 'px';
            tooltip.style.top = (rect.top - 30) + 'px';
        }}

        function hideTooltip() {{
            if (tooltip) {{
                tooltip.style.display = 'none';
            }}
        }}

        // Initial render
        renderText();
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding probe predictions")
    parser.add_argument("--probe_path", type=str, default="models/probe_embeddings.pth",
                        help="Path to probe file")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (will use from probe metadata if not specified)")
    parser.add_argument("--model_dtype", type=str, default="auto")
    parser.add_argument("--num_tokens", type=int, default=4000,
                        help="Number of tokens to visualize")
    parser.add_argument("--text_file", type=str, default="generated_text.txt",
                        help="Path to text file to visualize")
    parser.add_argument("--output", type=str, default="visualization_embeddings.html",
                        help="Output HTML file path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load probe
    print(f"Loading probe from {args.probe_path}...")
    probe, metadata = load_probe(args.probe_path, args.device)

    # Determine model name
    model_name = args.model_name or metadata["model_name"]

    # Load model
    dtype = get_model_dtype(args.model_dtype)
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get text to visualize
    text_path = Path(args.text_file)
    
    if not text_path.exists():
        pattern = f"{text_path.stem}*{text_path.suffix}"
        candidates = sorted(text_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime)
        
        if candidates:
            text_path = candidates[-1]
            print(f"Default text file '{args.text_file}' not found. Using latest match: {text_path}")
        else:
            raise FileNotFoundError(f"Text file not found: {args.text_file}")

    print(f"Loading text from {text_path}...")
    with open(text_path, 'r') as f:
        text = f.read()

    # Limit to num_tokens
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:args.num_tokens]
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    print(f"Generating predictions for {len(tokens)} tokens...")

    # Generate predictions
    predictions = generate_predictions(
        model,
        probe,
        token_ids,
        args.device,
    )

    # Create visualization
    create_html_visualization(
        text,
        tokens,
        predictions,
        metadata,
        args.output,
    )

    print(f"\nDone! Open {args.output} in a web browser to view the visualization.")


if __name__ == "__main__":
    main()
