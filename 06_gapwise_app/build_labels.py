"""Screen the full corpus, train a local student on separate AI-labelled examples,
and evaluate dictionary/rules/teacher/student against the existing human benchmark.
No benchmark labels or duplicate benchmark texts are allowed into student training.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score

from data import OUTPUT, load_jobs, load_courses, load_human
from taxonomy import rule_review, text_hash, LABELS, VERSION, SYSTEM_PROMPT
from deepseek_client import CACHE


def teacher_rows():
    if not CACHE.exists():
        return pd.DataFrame()
    records = []
    for line in CACHE.read_text().splitlines():
        row = json.loads(line)
        if row.get("prompt_hash") != text_hash(SYSTEM_PROMPT) or row.get("taxonomy_version") != VERSION:
            continue
        records.append({**{k: row[k] for k in ["record_id", "text_hash", "kind"]},
                        "teacher_label": row["review"]["label"],
                        "teacher_requirement_strength": row["review"]["requirement_strength"],
                        "teacher_maturity": row["review"]["maturity"],
                        "teacher_evidence": json.dumps(row["review"]["evidence_quotes"]),
                        "teacher_review": json.dumps(row["review"], ensure_ascii=False)})
    return pd.DataFrame(records).drop_duplicates(["kind", "text_hash"], keep="last") if records else pd.DataFrame()


def metrics(name, y, predicted, weights):
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", sample_weight=weights, zero_division=0)
    return {"model": name, "precision": precision, "recall": recall, "f1": f1,
            "accuracy": np.average(np.asarray(y)==np.asarray(predicted), weights=weights),
            "confusion_weighted": confusion_matrix(y, predicted, sample_weight=weights, labels=[0, 1]).tolist(),
            "n_reviewed": len(y)}


def screen(frame, kind):
    print(f"Local evidence screening: {kind} {len(frame):,}", flush=True)
    rows = []
    for r in frame[["record_id", "review_text", "text_hash", "AI Term Count", "group_hash"]].itertuples(index=False, name=None):
        identity, text, digest, terms, group = r
        review = rule_review(text)
        rows.append({"record_id": identity, "text_hash": digest, "group_hash": group, "kind": kind,
                     "legacy_ai": bool(terms > 0), "rule_label": review["label"],
                     "rule_maturity": review["maturity"],
                     "rule_evidence": json.dumps(review["evidence_quotes"], ensure_ascii=False),
                     "adoption_tasks": "; ".join(o["task"] for o in review["adoption_opportunities"]),
                     "adoption_evidence": json.dumps(review["adoption_opportunities"], ensure_ascii=False)})
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    jobs = load_jobs()
    human = load_human()
    # Exact group identities exclude duplicated text, not merely identical Job IDs.
    benchmark = jobs[jobs.record_id.isin(human.job_id)]
    benchmark_hashes = set(benchmark.text_hash)
    benchmark_groups = set(benchmark.group_hash)
    teacher = teacher_rows()
    job_labels = screen(jobs, "job")
    cet, uni = load_courses()
    course_labels = pd.concat([screen(cet, "course"), screen(uni, "course")], ignore_index=True)
    if not teacher.empty:
        fields = ["text_hash", "teacher_label", "teacher_maturity", "teacher_evidence", "teacher_review", "teacher_requirement_strength"]
        job_labels = job_labels.merge(teacher[teacher.kind.eq("job")][fields], on="text_hash", how="left", validate="many_to_one")
        course_labels = course_labels.merge(teacher[teacher.kind.eq("course")][fields], on="text_hash", how="left", validate="many_to_one")
    else:
        for frame in (job_labels, course_labels):
            for col in ["teacher_label", "teacher_maturity", "teacher_evidence", "teacher_review"]:
                frame[col] = np.nan

    evaluation = human.merge(job_labels, left_on="job_id", right_on="record_id", validate="one_to_one")
    if len(evaluation) != 400:
        raise ValueError("All 400 benchmark IDs must match the corpus")
    # Original validation strata must be retained, NOT recomputed with different tier rules.
    populations = {"High": 2746, "Probable": 1369, "Weak": 2695, "non-AI": 87506}
    ns = evaluation.classifier_tier.value_counts()
    evaluation["weight"] = evaluation.classifier_tier.map({k: v/ns[k] for k, v in populations.items()})
    y, weights = evaluation.is_AI_true.astype(int), evaluation.weight
    results = [metrics("dictionary_original", y, evaluation.legacy_ai, weights),
               metrics("rules_strict", y, evaluation.rule_label.eq("explicit"), weights),
               metrics("rules_expanded", y, evaluation.rule_label.isin(LABELS[:2]), weights)]
    tmask = evaluation.teacher_label.notna()
    if tmask.any():
        for name, positive in [("teacher_strict", ["explicit"]), ("teacher_expanded", list(LABELS[:2]))]:
            results.append(metrics(name, y[tmask], evaluation.loc[tmask, "teacher_label"].isin(positive), weights[tmask]))

    student_info = {"status": "not_trained", "reason": "At least 100 separate teacher-labelled examples and both binary classes required"}
    if not teacher.empty:
        train = teacher[teacher.kind.eq("job") & ~teacher.text_hash.isin(benchmark_hashes)].merge(
            jobs[["text_hash", "review_text", "AI Term Count", "group_hash"]].drop_duplicates("text_hash"), on="text_hash", validate="one_to_one")
        train = train[~train.group_hash.isin(benchmark_groups)].copy()
        train["target"] = train.teacher_label.isin(LABELS[:2]).astype(int)
        conflict_groups = train.groupby("group_hash").target.nunique()
        # Identical descriptions with conflicting teacher labels are excluded, not
        # resolved by silently taking whichever request completed last.
        train = train[~train.group_hash.isin(conflict_groups[conflict_groups.gt(1)].index)]
        train = train.drop_duplicates("group_hash").reset_index(drop=True)
        if len(train) >= 100 and train.target.value_counts().min() >= 10 and train.target.nunique() == 2:
            tr, dev = train_test_split(np.arange(len(train)), test_size=.25, random_state=588, stratify=train.target)
            # A binary model for observed demand; explicit-vs-implied is NOT guessed by this student.
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True, strip_accents="unicode")
            x = vec.fit_transform(train.iloc[tr].review_text)
            clf = LogisticRegression(C=2, max_iter=1500, class_weight="balanced", random_state=588)
            clf.fit(x, train.iloc[tr].target)
            dev_prob = clf.predict_proba(vec.transform(train.iloc[dev].review_text))[:, 1]
            # Tune only on independent AI-labelled development data; human benchmark stays untouched.
            choices = []
            for threshold in np.arange(.2, .81, .05):
                m = metrics("dev", train.iloc[dev].target, dev_prob>=threshold, np.ones(len(dev)))
                choices.append((m["f1"], float(threshold)))
            threshold = max(choices, key=lambda x: (x[0], -abs(x[1]-.5)))[1]
            dev_result = metrics("teacher_dev", train.iloc[dev].target, dev_prob>=threshold, np.ones(len(dev)))
            dictionary_dev = metrics("dictionary_on_teacher_dev", train.iloc[dev].target,
                                     train.iloc[dev]["AI Term Count"].gt(0), np.ones(len(dev)))
            # Keep model fitted on training only: development examples never become training examples.
            ptest = clf.predict_proba(vec.transform(benchmark.set_index("record_id").loc[evaluation.record_id].review_text))[:, 1]
            evaluation["student_score"] = ptest
            results.append(metrics("student_expanded", y, ptest>=threshold, weights))
            all_probs = []
            for start in range(0, len(jobs), 3000):
                all_probs.extend(clf.predict_proba(vec.transform(jobs.iloc[start:start+3000].review_text))[:, 1])
            job_labels["student_score"] = all_probs
            job_labels["student_ai"] = job_labels.student_score.ge(threshold)
            job_labels["student_uncertain"] = job_labels.student_score.between(threshold-.1, threshold+.1)
            with (OUTPUT / "student_model.pkl").open("wb") as stream:
                pickle.dump({"vectorizer": vec, "classifier": clf, "threshold": threshold, "taxonomy": VERSION}, stream)
            train.assign(split=np.where(train.index.isin(tr), "train", "development"))[
                ["record_id", "text_hash", "group_hash", "teacher_label", "target", "split"]].to_csv(OUTPUT / "student_split.csv", index=False)
            student_info = {"status": "trained", "n_training": len(tr), "n_development": len(dev),
                            "n_benchmark": len(evaluation), "threshold": threshold,
                            "development_agreement_with_teacher": dev_result,
                            "dictionary_agreement_with_teacher_development": dictionary_dev,
                            "benchmark_text_overlap": 0,
                            "benchmark_description_group_overlap": 0,
                            "conflicting_description_groups_excluded": int(conflict_groups.gt(1).sum()),
                            "score_warning": "Uncalibrated model scores, not probabilities of correctness.",
                            "scope": "Binary expanded demand only; no inferred explicit/implied split or maturity."}
    if "student_ai" not in job_labels:
        job_labels["student_score"] = np.nan
        job_labels["student_ai"] = False
        job_labels["student_uncertain"] = True
    # Teacher overlay is kept separate; it must NOT inflate student benchmark performance.
    job_labels["reviewed_by_llm"] = job_labels.teacher_label.notna()
    job_labels["expanded_ai"] = job_labels.rule_label.isin(LABELS[:2])
    job_labels["strict_ai"] = job_labels.rule_label.eq("explicit")
    for frame in (job_labels, course_labels):
        frame["screening_source"] = "local_rules"
        frame["evidence_label"] = frame.rule_label
        frame["maturity"] = frame.rule_maturity
        mask = frame.teacher_label.notna()
        frame.loc[mask, "screening_source"] = "deepseek"
        frame.loc[mask, "evidence_label"] = frame.loc[mask, "teacher_label"]
        frame.loc[mask, "maturity"] = frame.loc[mask, "teacher_maturity"]
        frame.loc[mask, "adoption_evidence"] = frame.loc[mask, "teacher_review"].map(
            lambda r: json.dumps(json.loads(r)["adoption_opportunities"], ensure_ascii=False))
        frame.loc[mask, "adoption_tasks"] = frame.loc[mask, "teacher_review"].map(
            lambda r: "; ".join(o["task"] for o in json.loads(r)["adoption_opportunities"]))
        frame["strict_ai"] = frame.evidence_label.eq("explicit")
        frame["expanded_ai"] = frame.evidence_label.isin(LABELS[:2])
    job_labels.to_csv(OUTPUT / "job_labels.csv", index=False)
    course_labels.to_csv(OUTPUT / "course_labels.csv", index=False)
    evaluation.to_csv(OUTPUT / "benchmark_predictions.csv", index=False)
    disagreement = evaluation[evaluation.teacher_label.notna() &
        (evaluation.is_AI_true.astype(bool) != evaluation.teacher_label.isin(LABELS[:2]))]
    disagreement.to_csv(OUTPUT / "benchmark_disagreements.csv", index=False)
    # These old "misses" are especially influential because each represents 875.06 jobs.
    broad_positives = evaluation[evaluation.classifier_tier.eq("non-AI") & evaluation.is_AI_true.eq(1)]
    (OUTPUT / "benchmark_construct_audit.json").write_text(json.dumps({
        "old_untagged_human_positive_count": len(broad_positives),
        "teacher_labels_for_old_untagged_positives": broad_positives.teacher_label.value_counts(dropna=False).to_dict(),
        "interpretation": "Existing human positives include generic software, desktop support and infrastructure jobs without observed AI requirements. Strict observed-demand and potential-adoption constructs differ. Preserve both labels; do not silently rewrite benchmark truth or treat low agreement as proof of either side's correctness.",
        "all_teacher_disagreements": len(disagreement)}, indent=2))
    (OUTPUT / "model_comparison.json").write_text(json.dumps({
        "results": results, "student": student_info,
        "benchmark_warning": "Existing one-reviewer labels, previously inspected; not a new blinded ground truth. AI agreement does not establish truth.",
        "population_weighting": populations,
        "review_coverage": {"jobs": len(job_labels), "jobs_llm_reviewed": int(job_labels.reviewed_by_llm.sum()),
                            "courses": len(course_labels), "courses_llm_reviewed": int(course_labels.teacher_label.notna().sum()),
                            "unique_job_texts_reviewed": int(teacher.kind.eq("job").sum()) if not teacher.empty else 0,
                            "unique_course_texts_reviewed": int(teacher.kind.eq("course").sum()) if not teacher.empty else 0,
                            "identical_text_propagation": True},
        "label_counts": job_labels.evidence_label.value_counts().to_dict(),
        "selection": {"default": "strict/expanded evidence-screening scenarios remain provisional",
                      "student_auto_promoted": False,
                      "reason": "No new classifier is automatically promoted based on teacher agreement; inspect original-benchmark tradeoffs and construct differences."},
        "taxonomy_version": VERSION, "prompt_hash": text_hash(SYSTEM_PROMPT)
    }, indent=2))
    print(json.dumps({"comparison": results, "student": student_info}, indent=2), flush=True)


if __name__ == "__main__":
    main()
