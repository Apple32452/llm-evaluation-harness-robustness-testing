import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICE_LABELS = ["A", "B", "C", "D"]


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt(row: Dict) -> str:
    choices = "\n".join(f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, row["choices"]))
    return (
        "Answer the multiple-choice question by selecting A, B, C, or D.\n\n"
        f"Question: {row['question']}\n"
        f"{choices}\n"
        "Answer:"
    )


@torch.no_grad()
def continuation_logprob(model, tokenizer, prompt: str, continuation: str, device: torch.device) -> float:
    full_text = prompt + " " + continuation
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    # Continuation tokens are the suffix after the prompt tokenization.
    start = prompt_ids.shape[1]
    if full_ids.shape[1] <= start:
        return float("-inf")

    logits = model(full_ids[:, :-1]).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    target_ids = full_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Tokens whose targets correspond to continuation.
    continuation_token_log_probs = token_log_probs[:, start - 1 :]
    return float(continuation_token_log_probs.sum().item())


def expected_calibration_error(confidences: List[float], correct: List[int], n_bins: int = 10) -> float:
    confidences_arr = np.array(confidences)
    correct_arr = np.array(correct)
    ece = 0.0

    for bin_idx in range(n_bins):
        lo, hi = bin_idx / n_bins, (bin_idx + 1) / n_bins
        mask = (confidences_arr > lo) & (confidences_arr <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(confidences_arr[mask].mean())
        bin_acc = float(correct_arr[mask].mean())
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)

    return ece


def evaluate(model_name: str, data: List[Dict], device: torch.device) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    records = []
    confidences = []
    correctness = []

    for row in tqdm(data, desc="evaluating"):
        prompt = build_prompt(row)
        logps = []
        for label in CHOICE_LABELS:
            lp = continuation_logprob(model, tokenizer, prompt, label, device)
            logps.append(lp)

        probs = np.exp(np.array(logps) - np.max(logps))
        probs = probs / probs.sum()

        pred_idx = int(np.argmax(probs))
        pred = CHOICE_LABELS[pred_idx]
        correct = int(pred == row["answer"])
        confidence = float(probs[pred_idx])

        confidences.append(confidence)
        correctness.append(correct)

        records.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "question": row["question"],
                "answer": row["answer"],
                "prediction": pred,
                "correct": bool(correct),
                "confidence": confidence,
                "choice_probabilities": {label: float(prob) for label, prob in zip(CHOICE_LABELS, probs)},
            }
        )

    accuracy = float(np.mean(correctness)) if correctness else 0.0
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    ece = expected_calibration_error(confidences, correctness) if correctness else 0.0
    nll = float(
        np.mean(
            [
                -math.log(max(rec["choice_probabilities"][rec["answer"]], 1e-12))
                for rec in records
            ]
        )
    ) if records else 0.0

    by_category = {}
    for cat in sorted({r["category"] for r in records}):
        subset = [r for r in records if r["category"] == cat]
        by_category[cat] = {
            "n": len(subset),
            "accuracy": float(np.mean([r["correct"] for r in subset])),
            "avg_confidence": float(np.mean([r["confidence"] for r in subset])),
        }

    return {
        "model": model_name,
        "num_examples": len(records),
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "expected_calibration_error": ece,
        "negative_log_likelihood": nll,
        "by_category": by_category,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results/eval.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    data = load_jsonl(args.data)
    result = evaluate(args.model, data, device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    print(f"Wrote full records to {out}")


if __name__ == "__main__":
    main()
