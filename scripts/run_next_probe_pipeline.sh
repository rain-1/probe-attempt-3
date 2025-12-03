#!/usr/bin/env bash
# End-to-end helper for preparing next-token probe data, uploading it to
# Hugging Face, and retraining the layer-0 probe. Adjust env vars as needed.

set -euo pipefail

# -------- Configuration (override via env vars before running) -------- #
MODEL_NAME=${MODEL_NAME:-"unsloth/gemma-3-270m"}
TEXT_FILE=${TEXT_FILE:-"generated_text_0.txt"}
TARGET_TOKEN=${TARGET_TOKEN:-" the"}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-256}
OUTPUT_DIR=${OUTPUT_DIR:-"data_model_next"}
BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-20}
HF_DATASET_ID=${HF_DATASET_ID:-""}           # e.g. username/gemma-next-probe
HF_UPLOAD_SUBDIR=${HF_UPLOAD_SUBDIR:-"next-probe"}  # folder inside HF repo
# --------------------------------------------------------------------- #

if [[ ! -f "$TEXT_FILE" ]]; then
  echo "[!] TEXT_FILE '$TEXT_FILE' not found. Generate text first." >&2
  exit 1
fi

echo "[1/4] Preparing next-token dataset from $TEXT_FILE"
python scripts/prepare_data_from_file.py \
  --input_file "$TEXT_FILE" \
  --model_name "$MODEL_NAME" \
  --target_token "$TARGET_TOKEN" \
  --context_length "$CONTEXT_LENGTH" \
  --task next \
  --output_dir "$OUTPUT_DIR"

TRAIN_PATH="$OUTPUT_DIR/train.pth"
VAL_PATH="$OUTPUT_DIR/val.pth"

if [[ ! -f "$TRAIN_PATH" || ! -f "$VAL_PATH" ]]; then
  echo "[!] Expected $TRAIN_PATH and $VAL_PATH after preparation." >&2
  exit 1
fi

echo "[2/4] (Optional) Uploading dataset artifacts to Hugging Face"
if [[ -n "$HF_DATASET_ID" ]]; then
  if ! command -v huggingface-cli >/dev/null; then
    echo "[!] huggingface-cli not found. Install via 'pip install huggingface-hub'." >&2
    exit 1
  fi
  if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "[!] Not logged in to Hugging Face. Run 'huggingface-cli login' first." >&2
    exit 1
  fi

  ARCHIVE_NAME="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S).tar.gz"
  tar -czf "$ARCHIVE_NAME" -C "$OUTPUT_DIR" .
  echo "    Uploading $ARCHIVE_NAME to hf://$HF_DATASET_ID/$HF_UPLOAD_SUBDIR/"
  huggingface-cli upload "$HF_DATASET_ID" "$ARCHIVE_NAME" \
    "$HF_UPLOAD_SUBDIR/$ARCHIVE_NAME" --repo-type dataset --private
  echo "    Upload complete. Remove local archive if desired."
else
  echo "    HF_DATASET_ID not set; skipping upload."
fi

echo "[3/4] Training layer-17 next-token probe"
python scripts/train_probe.py \
  --train_data "$TRAIN_PATH" \
  --val_data "$VAL_PATH" \
  --layer 17 \
  --task next \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --no_wandb

MODEL_PATH="models/probe_layer17_next.pth"
if [[ -f "$MODEL_PATH" ]]; then
  echo "    Probe saved to $MODEL_PATH"
fi

echo "[4/4] (Optional) Visualize the trained probe"
python scripts/visualize_probe.py \
  --probe_dir models \
  --text_source file \
  --text_file "$TEXT_FILE" \
  --num_tokens 2000 \
  --output visualization_next.html

echo "Pipeline complete. Adjust NUM tokens or layers as needed."
