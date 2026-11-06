"""Paired, stratified bootstrap on the unchanged, reused 400-job benchmark."""
import json
import numpy as np
import pandas as pd
from data import OUTPUT


def binary_scores(y, p, w):
    tp = w[(y==1) & p].sum()
    fp = w[(y==0) & p].sum()
    fn = w[(y==1) & ~p].sum()
    return np.array([tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0,
                     2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0])


def main():
    d = pd.read_csv(OUTPUT / "benchmark_predictions.csv")
    info = json.loads((OUTPUT / "model_comparison.json").read_text())
    threshold = info["student"]["threshold"]
    y, old, new, w = d.is_AI_true.to_numpy(), d.legacy_ai.to_numpy(bool), d.student_score.ge(threshold).to_numpy(), d.weight.to_numpy()
    groups = [np.flatnonzero(d.classifier_tier.to_numpy()==tier) for tier in d.classifier_tier.unique()]
    rng = np.random.default_rng(588)
    previous, student = [], []
    for _ in range(2000):
        index = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        previous.append(binary_scores(y[index], old[index], w[index]))
        student.append(binary_scores(y[index], new[index], w[index]))
    previous, student = np.array(previous), np.array(student)
    difference = student-previous
    result = {"n": len(d), "replicates": 2000, "method": "paired bootstrap within original sampling strata",
              "metrics_order": ["precision", "recall", "f1"],
              "dictionary_point": binary_scores(y, old, w).tolist(),
              "student_point": binary_scores(y, new, w).tolist(),
              "dictionary_95_percentile_interval": np.quantile(previous, [.025, .975], axis=0).tolist(),
              "student_95_percentile_interval": np.quantile(student, [.025, .975], axis=0).tolist(),
              "paired_difference_95_percentile_interval": np.quantile(difference, [.025, .975], axis=0).tolist(),
              "warning": "Conditional on the existing one-reviewer labels and reused benchmark; not uncertainty over label truth or an independent final test."}
    (OUTPUT / "benchmark_uncertainty.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
