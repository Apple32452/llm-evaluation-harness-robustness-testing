# LLM Evaluation Harness & Robustness Testing

A reproducible evaluation project for measuring reasoning accuracy, calibration, prompt perturbation robustness, and failure modes for open-source language models.

This project is designed to support a Meta Research Engineer / GenAI Evaluation resume claim with concrete scripts and outputs.

## What this project demonstrates

- Multiple-choice log-likelihood evaluation for causal language models
- Accuracy, negative log-likelihood, confidence, and expected calibration error
- Prompt perturbation testing across equivalent prompt formats
- Failure-mode clustering using TF-IDF and KMeans
- Optional LoRA/QLoRA fine-tuning skeleton
- Optional integration notes for EleutherAI lm-evaluation-harness

## Repository structure

```text
llm-evaluation-harness-robustness-testing/
├── data/
│   └── eval_questions.jsonl
├── scripts/
│   └── run_lm_eval_example.sh
├── src/
│   ├── evaluate_mcq.py
│   ├── perturb_prompts.py
│   ├── cluster_failures.py
│   └── train_lora_skeleton.py
├── results/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run a lightweight evaluation

Use a tiny model first to validate the pipeline:

```bash
python src/evaluate_mcq.py \
  --model sshleifer/tiny-gpt2 \
  --data data/eval_questions.jsonl \
  --out results/tiny_gpt2_eval.json
```

Use a stronger open model when you have enough compute:

```bash
python src/evaluate_mcq.py \
  --model distilgpt2 \
  --data data/eval_questions.jsonl \
  --out results/distilgpt2_eval.json
```

## Run prompt perturbation robustness testing

```bash
python src/perturb_prompts.py \
  --data data/eval_questions.jsonl \
  --out data/eval_questions_perturbed.jsonl

python src/evaluate_mcq.py \
  --model distilgpt2 \
  --data data/eval_questions_perturbed.jsonl \
  --out results/distilgpt2_perturbed_eval.json
```

## Cluster model failures

```bash
python src/cluster_failures.py \
  --eval-json results/distilgpt2_eval.json \
  --out results/failure_clusters.json
```

## Optional lm-evaluation-harness example

```bash
bash scripts/run_lm_eval_example.sh
```

## Optional LoRA / QLoRA training skeleton

The `src/train_lora_skeleton.py` file shows how to structure PEFT fine-tuning. Treat it as a template and plug in a real instruction dataset before using results on a resume.

## Resume bullets after running experiments

Use only after you run the pipeline and replace placeholders with your measured numbers:

- Built a reproducible LLM evaluation harness for multiple-choice reasoning, calibration, prompt perturbation robustness, and failure-mode clustering across open-source transformer models.
- Implemented log-likelihood scoring, expected calibration error, and prompt-variant stress tests to identify brittle reasoning and hallucination-prone failure modes.
- Automated evaluation artifacts with fixed seeds, JSON outputs, dataset versioning, and optional lm-evaluation-harness integration for benchmark-style reporting.

## Notes

This project is intentionally honest: it gives you a real evaluation pipeline, but your final resume metrics should come from experiments you actually run.
