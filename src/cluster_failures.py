import argparse
import json
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--out", default="results/failure_clusters.json")
    parser.add_argument("--clusters", type=int, default=3)
    args = parser.parse_args()

    data = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    failures = [r for r in data["records"] if not r["correct"]]

    if not failures:
        result = {"message": "No failures to cluster.", "clusters": []}
    else:
        texts = [f'{r["category"]}: {r["question"]}' for r in failures]
        n_clusters = min(args.clusters, len(texts))
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        model = KMeans(n_clusters=n_clusters, random_state=7, n_init="auto")
        labels = model.fit_predict(X)

        terms = vectorizer.get_feature_names_out()
        clusters = []
        for cluster_id in range(n_clusters):
            idxs = [i for i, label in enumerate(labels) if label == cluster_id]
            center = model.cluster_centers_[cluster_id]
            top_terms = [terms[i] for i in center.argsort()[-8:][::-1]]
            examples = [failures[i] for i in idxs]
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(examples),
                    "top_terms": top_terms,
                    "examples": [
                        {
                            "id": ex["id"],
                            "category": ex["category"],
                            "question": ex["question"],
                            "answer": ex["answer"],
                            "prediction": ex["prediction"],
                            "confidence": ex["confidence"],
                        }
                        for ex in examples[:5]
                    ],
                }
            )
        result = {"num_failures": len(failures), "clusters": clusters}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
