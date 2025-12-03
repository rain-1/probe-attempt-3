#!/usr/bin/env python3
"""
Generate interactive HTML visualization of probe predictions.
Shows probe predictions across different layers with a slider.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


class LinearProbe(nn.Module):
    """Simple linear probe for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def load_probe(probe_path: str, device: str):
    """Load probe from file with metadata."""
    data = torch.load(probe_path, map_location=device)
    metadata = data["metadata"]

    probe = LinearProbe(metadata["hidden_size"]).to(device)
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


def extract_hidden_states(model, input_ids: torch.Tensor, layer: int):
    """Extract hidden states from a specific layer."""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
    return hidden_states


def generate_predictions(
    model,
    tokenizer,
    probes: Dict[int, nn.Module],
    text: str,
    device: str,
) -> Dict[int, List[float]]:
    """
    Generate predictions for each token in the text using probes from different layers.
    Returns dict mapping layer -> list of probabilities (one per token).
    """
    # Tokenize
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([token_ids], device=device)

    # Get predictions for each layer
    layer_predictions = {}

    for layer, probe in probes.items():
        # Extract hidden states for this layer
        hidden_states = extract_hidden_states(model, input_ids, layer)  # (1, seq_len, hidden_dim)

        # Get prediction for each token
        predictions = []
        for i in range(hidden_states.shape[1]):
            hidden = hidden_states[0, i, :].unsqueeze(0).float()  # (1, hidden_dim)

            # Check for NaN
            if torch.isnan(hidden).any():
                predictions.append(0.0)
                continue

            logit = probe(hidden).squeeze(-1)
            prob = torch.sigmoid(logit).item()

            # Convert NaN to 0 for JSON serialization
            if prob != prob:  # NaN check
                prob = 0.0

            predictions.append(prob)

        layer_predictions[layer] = predictions

    return layer_predictions


def create_html_visualization(
    text: str,
    tokens: List[str],
    layer_predictions: Dict[int, List[float]],
    metadata: dict,
    output_path: str,
):
    """Create interactive HTML visualization."""

    # Prepare data for JavaScript (convert NaN to null for JSON)
    layers = sorted(layer_predictions.keys())
    predictions_data = {}
    for layer in layers:
        preds = layer_predictions[layer]
        # Convert any remaining NaN to None (null in JSON)
        preds_clean = [None if (p != p) else p for p in preds]
        predictions_data[str(layer)] = preds_clean

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Probe Visualization - Layer Sweep</title>
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

        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .slider-container {{
            margin: 20px 0;
        }}

        #layerSlider {{
            width: 100%;
            height: 40px;
        }}

        .slider-label {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            text-align: center;
            margin-bottom: 10px;
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
    </style>
</head>
<body>
    <h1>Probe Prediction Visualization</h1>

    <div class="metadata">
        <p><strong>Model:</strong> {metadata.get('model_name', 'N/A')}</p>
        <p><strong>Target Token:</strong> "{metadata.get('target_token', 'N/A')}"</p>
        <p><strong>Task:</strong> {"Current token" if metadata.get('task') == 'current' else "Next token"} prediction</p>
        <p><strong>Layers:</strong> {min(layers)} - {max(layers)}</p>
    </div>

    <div class="controls">
        <div class="slider-label">Layer: <span id="currentLayer">{layers[0]}</span></div>
        <div class="slider-container">
            <input type="range" id="layerSlider" min="0" max="{len(layers)-1}" value="0" step="1">
        </div>
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
    </div>

    <script>
        const tokens = {json.dumps(tokens)};
        const predictions = {json.dumps(predictions_data)};
        const layers = {json.dumps(layers)};

        const slider = document.getElementById('layerSlider');
        const currentLayerSpan = document.getElementById('currentLayer');
        const textContainer = document.getElementById('textContainer');

        let tooltip = null;

        function getColorForProbability(prob) {{
            if (prob === null || prob === undefined || isNaN(prob)) {{
                return 'rgba(128, 128, 128, 0.2)';  // Gray for invalid
            }}

            const alpha = Math.abs(prob - 0.5) * 2;  // 0 at 0.5, 1 at 0 or 1
            const red = Math.round(255 * prob);
            const green = Math.round(255 * (1 - prob));
            const blue = Math.round(255 * (1 - prob));

            return `rgba(${{red}}, ${{green}}, ${{blue}}, ${{alpha * 0.6 + 0.1}})`;
        }}

        function renderText() {{
            const layerIndex = parseInt(slider.value);
            const layer = layers[layerIndex];
            currentLayerSpan.textContent = layer;

            const layerPredictions = predictions[layer.toString()];

            let html = '';
            for (let i = 0; i < tokens.length; i++) {{
                const token = tokens[i];
                const prob = layerPredictions[i];
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

        slider.addEventListener('input', renderText);

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
    parser = argparse.ArgumentParser(description="Visualize probe predictions")
    parser.add_argument("--probe_dir", type=str, default="models",
                        help="Directory containing probe files")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (will use from probe metadata if not specified)")
    parser.add_argument("--model_dtype", type=str, default="auto")
    parser.add_argument("--num_tokens", type=int, default=4000,
                        help="Number of tokens to visualize")
    parser.add_argument("--text_source", type=str, default="gutenberg",
                        choices=["gutenberg", "file"],
                        help="Source of text to visualize")
    parser.add_argument("--text_file", type=str, default=None,
                        help="Path to text file (if text_source=file)")
    parser.add_argument("--language", type=str, default="en",
                        help="Language split to use for Gutenberg (en, fr, de, es, etc.)")
    parser.add_argument("--output", type=str, default="visualization.html",
                        help="Output HTML file path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Find all probe files
    probe_dir = Path(args.probe_dir)
    probe_files = sorted(probe_dir.glob("probe_layer*.pth"))

    if not probe_files:
        print(f"No probe files found in {args.probe_dir}")
        return

    print(f"Found {len(probe_files)} probe files")

    # Load probes
    probes = {}
    metadata = None

    for probe_file in probe_files:
        probe, meta = load_probe(str(probe_file), args.device)
        layer = meta["layer"]
        probes[layer] = probe

        if metadata is None:
            metadata = meta

        print(f"Loaded probe for layer {layer}")

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
    if args.text_source == "file" and args.text_file:
        print(f"Loading text from {args.text_file}...")
        with open(args.text_file, 'r') as f:
            text = f.read()
    else:
        print(f"Loading random text from Project Gutenberg (language: {args.language})...")
        dataset = load_dataset("manu/project_gutenberg", split=args.language, streaming=True)
        for item in dataset:
            if item.get("text") and len(item["text"]) > 1000:
                text = item["text"]
                break

    # Limit to num_tokens
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:args.num_tokens]
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    text = tokenizer.decode(token_ids)

    print(f"Generating predictions for {len(tokens)} tokens across {len(probes)} layers...")

    # Generate predictions
    layer_predictions = generate_predictions(
        model,
        tokenizer,
        probes,
        text,
        args.device,
    )

    # Create visualization
    create_html_visualization(
        text,
        tokens,
        layer_predictions,
        metadata,
        args.output,
    )

    print(f"\nDone! Open {args.output} in a web browser to view the visualization.")


if __name__ == "__main__":
    main()
