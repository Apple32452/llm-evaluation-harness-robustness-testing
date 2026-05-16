#!/usr/bin/env bash
set -euo pipefail

# Install separately if needed:
# pip install git+https://github.com/EleutherAI/lm-evaluation-harness

lm_eval \
  --model hf \
  --model_args pretrained=distilgpt2 \
  --tasks hellaswag \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path results/lm_eval_distilgpt2_hellaswag.json
