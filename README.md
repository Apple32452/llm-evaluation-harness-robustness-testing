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

# LLM Evaluation Harness & Robustness Testing

## Summary

This project builds a reproducible LLM evaluation harness for multiple-choice reasoning, calibration analysis, and prompt-robustness testing. The pipeline evaluates a Hugging Face causal language model, computes accuracy, confidence, expected calibration error, negative log-likelihood, and category-level performance, then tests robustness by evaluating perturbed versions of the same prompts.

Using `sshleifer/tiny-gpt2` as a fast sanity-check model, the system successfully evaluated 8 original examples and 40 perturbed examples. The model achieved 0% accuracy on the original benchmark and 5% accuracy on the perturbed benchmark, which is expected for a very small model. The main contribution is the evaluation infrastructure: a reusable pipeline for measuring model correctness, confidence, calibration, robustness, and failure modes.

## Results

The evaluation pipeline was tested using `sshleifer/tiny-gpt2`, a very small GPT-2 model intended for fast sanity checks rather than strong reasoning performance.

### Environment Setup

The project environment was successfully created using a Python virtual environment. The required packages were installed, including:

- `torch`
- `transformers`
- `accelerate`
- `scikit-learn`
- `peft`
- `numpy`
- `tqdm`

On macOS, `bitsandbytes` was skipped automatically because it is not supported on Darwin/macOS in this setup. This did not affect the core evaluation pipeline.

### Base Multiple-Choice Evaluation

The model was first evaluated on the original multiple-choice benchmark containing 8 examples across four categories:

- Math
- Machine Learning
- Reasoning
- Systems

Command used:

```bash
python src/evaluate_mcq.py \
  --model sshleifer/tiny-gpt2 \
  --data data/eval_questions.jsonl \
  --out results/tiny_gpt2_eval.json
```

The evaluation completed successfully and produced the following summary:

| Metric | Value |
|---|---:|
| Model | `sshleifer/tiny-gpt2` |
| Number of Examples | 8 |
| Accuracy | 0.00 |
| Average Confidence | 0.2559 |
| Expected Calibration Error | 0.2559 |
| Negative Log-Likelihood | 1.3897 |

Category-level performance:

| Category | Examples | Accuracy | Average Confidence |
|---|---:|---:|---:|
| Math | 2 | 0.00 | 0.2563 |
| Machine Learning | 2 | 0.00 | 0.2561 |
| Reasoning | 2 | 0.00 | 0.2569 |
| Systems | 2 | 0.00 | 0.2544 |

The model achieved 0 out of 8 correct answers. This is expected because `sshleifer/tiny-gpt2` is an extremely small model used only to verify that the evaluation harness runs correctly.

### Prompt Perturbation Robustness Test

Next, the benchmark was expanded using prompt perturbations. The perturbation script generated 40 perturbed examples from the original 8-question dataset.

Command used:

```bash
python src/perturb_prompts.py \
  --data data/eval_questions.jsonl \
  --out data/eval_questions_perturbed.jsonl
```

Output:

```bash
Wrote 40 perturbed examples to data/eval_questions_perturbed.jsonl
```

The perturbed benchmark was then evaluated with the same model:

```bash
python src/evaluate_mcq.py \
  --model sshleifer/tiny-gpt2 \
  --data data/eval_questions_perturbed.jsonl \
  --out results/tiny_gpt2_perturbed_eval.json
```

The evaluation completed successfully and produced the following summary:

| Metric | Value |
|---|---:|
| Model | `sshleifer/tiny-gpt2` |
| Number of Examples | 40 |
| Accuracy | 0.05 |
| Average Confidence | 0.2563 |
| Expected Calibration Error | 0.2063 |
| Negative Log-Likelihood | 1.3890 |

Category-level performance:

| Category | Examples | Accuracy | Average Confidence |
|---|---:|---:|---:|
| Math | 10 | 0.00 | 0.2564 |
| Machine Learning | 10 | 0.10 | 0.2564 |
| Reasoning | 10 | 0.00 | 0.2565 |
| Systems | 10 | 0.10 | 0.2558 |

The model achieved 5% accuracy on the perturbed dataset. This shows that the evaluation system can measure robustness under prompt variations, even though the tiny test model itself performs poorly.

## Interpretation

The goal of this experiment was not to demonstrate strong language-model performance. Instead, the goal was to verify that the evaluation harness can run end-to-end and produce useful diagnostic metrics.

The pipeline successfully supports:

- Multiple-choice model evaluation
- Accuracy measurement
- Confidence estimation
- Expected Calibration Error calculation
- Negative Log-Likelihood calculation
- Category-level performance breakdown
- Prompt perturbation robustness testing
- JSON output for reproducible experiment tracking

The results show that `sshleifer/tiny-gpt2` behaves close to random guessing on this benchmark. Its confidence scores are around 0.25, which is expected for a four-choice multiple-choice setting where each answer has roughly equal probability.

The low accuracy is therefore not a failure of the code. It confirms that the harness can expose weak reasoning, poor task performance, and limited robustness in small language models.

## Conclusion

This project implements a lightweight but extensible LLM evaluation harness for testing reasoning, factuality, calibration, and robustness. The system evaluates language models on multiple-choice tasks, records model confidence, computes calibration metrics, and measures performance under prompt perturbations.

In the initial experiment, `sshleifer/tiny-gpt2` achieved 0% accuracy on the original 8-question benchmark and 5% accuracy on the 40-example perturbed benchmark. These results are expected because the model is intentionally small and was used only as a fast sanity-check model.

The key result is that the evaluation infrastructure works successfully end-to-end. It can be extended to larger models such as `distilgpt2`, instruction-tuned models, or Hugging Face causal language models. Future work includes evaluating stronger models, adding larger benchmark datasets, comparing base and fine-tuned models, adding pairwise preference evaluation, and improving failure-mode clustering.

Overall, this project demonstrates reproducible LLM evaluation, prompt-robustness testing, calibration analysis, and failure-mode diagnostics in a compact research-engineering pipeline.

## Next Step: Failure Clustering

The failure clustering output was not generated in the initial run because the clustering script had not yet been executed. To generate it, run:

```bash
python src/cluster_failures.py \
  --eval-json results/tiny_gpt2_eval.json \
  --out results/failure_clusters.json
```

Then inspect the result:

```bash
cat results/failure_clusters.json
```

