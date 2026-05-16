import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def perturb(row: Dict) -> Iterable[Dict]:
    base = dict(row)

    variants = {
        "original": row["question"],
        "polite": "Please answer carefully: " + row["question"],
        "concise": row["question"] + " Choose the best answer.",
        "uppercase": row["question"].upper(),
        "robust_instruction": "Think about the question, but output only the correct option letter. " + row["question"],
    }

    for name, question in variants.items():
        new_row = dict(base)
        new_row["id"] = f"{row.get('id', 'item')}_{name}"
        new_row["perturbation"] = name
        new_row["question"] = question
        yield new_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    out_rows = []
    for row in rows:
        out_rows.extend(list(perturb(row)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(out_rows)} perturbed examples to {out}")


if __name__ == "__main__":
    main()
