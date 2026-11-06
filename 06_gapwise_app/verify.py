"""Fail loudly on missing metrics, source drift, train/test overlap or incomplete runs."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data import PROJECT, OUTPUT, ROOT, load_jobs, load_courses
from taxonomy import VERSION, SYSTEM_PROMPT, text_hash, validate_review
from course_designer import design

COVERAGE = {
    "data_quality": "tbl_audit.csv",
    "confidence_tiers": "tier.csv",
    "sector_demand_penetration_enrichment": "cat_stats.csv",
    "career_stage": "pos_stats.csv",
    "experience_barriers": "experience_barriers.csv",
    "overall_rates_hhi_top5_top10": "headline_metrics.csv",
    "maturity_by_sector": "hm_norm.csv",
    "skill_demand": "skill_demand.csv",
    "skill_cooccurrence_jaccard_lift": "edges.csv",
    "occupation_retention_reskilling": "fam_tbl.csv",
    "reskilling_opportunity": "mtx.csv",
    "education_accessibility": "acc.csv",
    "salary_sensitivity_models": "salary_models.csv",
    "provider_concentration": "cet.csv",
    "course_registration_cohorts": "cy.csv",
    "course_outcomes": "imp.csv",
    "university_length_adjustment": "res.csv",
    "university_location_quotients": "lq.csv",
    "university_theme_diversity": "diversity.csv",
    "nmf_topics": "topics.csv",
    "course_crowding": "courses_ai.csv",
    "semantic_alignment": "align.csv",
    "employer_curriculum_skill_gaps": "comp.csv",
    "sector_opportunity_components": "sec.csv",
    "weight_sensitivity_200_draws": "sens.csv",
    "course_candidate_ranking": "candidate_scores_v4.csv",
    "job_course_skill_ratios": "skill_mention_gaps.csv",
    "sector_course_ratios": "sector_course_ratios.csv",
    "bundle_depth": "bundle_depth.csv",
    "job_course_maturity_comparison": "job_course_maturity.csv",
    "potential_adoption_separate": "adoption_opportunities.csv",
    "potential_adoption_task_quotes": "adoption_task_evidence.csv",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    jl = pd.read_csv(OUTPUT / "job_labels.csv")
    cl = pd.read_csv(OUTPUT / "course_labels.csv")
    bench = pd.read_csv(OUTPUT / "benchmark_predictions.csv")
    split = pd.read_csv(OUTPUT / "student_split.csv")
    comparison = json.loads((OUTPUT / "model_comparison.json").read_text())
    checks = {
        "job_count_94316": len(jl)==94316,
        "course_count_52175": len(cl)==52175,
        "unique_job_ids": jl.record_id.is_unique,
        "unique_course_ids": cl.record_id.is_unique,
        "benchmark_400_unique": len(bench)==400 and bench.job_id.is_unique,
        "no_benchmark_training_text_overlap": not bool(set(bench.text_hash) & set(split.text_hash)),
        "no_benchmark_training_description_overlap": not bool(set(bench.group_hash) & set(split.group_hash)),
        "no_training_development_text_overlap": not bool(set(split[split.split.eq("train")].text_hash) & set(split[split.split.eq("development")].text_hash)),
        "no_training_development_description_overlap": not bool(set(split[split.split.eq("train")].group_hash) & set(split[split.split.eq("development")].group_hash)),
        "complete_scenario_flags": not jl[["legacy_ai", "strict_ai", "expanded_ai", "student_ai"]].isna().any().any(),
        "strict_subset_expanded": not bool((jl.strict_ai & ~jl.expanded_ai).any()),
        "current_taxonomy": comparison["taxonomy_version"]==VERSION,
        "current_prompt": comparison["prompt_hash"]==text_hash(SYSTEM_PROMPT),
    }
    bad_quotes, quote_count = [], 0
    for frames, labels in [([load_jobs()], jl), (load_courses(), cl)]:
        source = pd.concat([f[['record_id', 'review_text']] for f in frames]).set_index('record_id').review_text
        for r in labels[labels.teacher_label.notna()].itertuples(index=False):
            quote_count += 1
            try:
                validate_review(json.loads(r.teacher_review), source.loc[r.record_id])
            except ValueError:
                bad_quotes.append(r.record_id)
    checks['all_reviewed_and_propagated_quotes_valid'] = not bad_quotes
    scenarios, summaries = {}, []
    for name in ["legacy", "strict", "expanded", "student"]:
        base = OUTPUT / "metrics" / name
        manifest = json.loads((base / "run_manifest.json").read_text())
        notebook = json.loads((base / "analysis_executed.ipynb").read_text())
        code = [c for c in notebook["cells"] if c["cell_type"]=="code"]
        checks[name+"_complete_34_cells"] = manifest["status"]=="complete" and len(manifest["completed_cells"])==34 and all(c["execution_count"] for c in code)
        checks[name+"_no_notebook_errors"] = not any(o["output_type"]=="error" for c in code for o in c.get("outputs", []))
        checks[name+"_figures_embedded_in_notebook"] = sum(bool(c.get('attachments')) for c in notebook['cells']) == len(manifest['figures'])
        checks[name+"_labels_match"] = manifest["job_labels_sha256"]==digest(OUTPUT / "job_labels.csv") and manifest["course_labels_sha256"]==digest(OUTPUT / "course_labels.csv")
        checks[name+"_source_matches"] = manifest["source_generator_sha256"]==digest(PROJECT / "05_analysis/build_company_x_eda.py")
        missing = [file for file in COVERAGE.values() if not (base / "tables" / file).exists()]
        checks[name+"_all_metric_families"] = not missing
        checks[name+"_figures_exist"] = len(manifest["figures"])>=25 and all((base / "figures" / f).exists() for f in manifest["figures"])
        scores = pd.read_csv(base / "tables/candidate_scores_v4.csv")
        checks[name+"_rank_matches_score"] = scores.score.is_monotonic_decreasing
        scenarios[name] = {"metric_families": len(COVERAGE), "missing": missing,
                           "figures": len(manifest["figures"]), "code_cells": len(code),
                           "top_candidate": scores.iloc[0].concept if len(scores) else None}
        summaries.append(json.loads((base / "summary.json").read_text()))
        draft = design("Finance", "Working finance professionals", name)
        checks[name+"_draft_evidence_exists"] = draft["status"]=="draft_requires_provider_review" and all(Path(p).exists() for p in draft["evidence"])
        checks[name+"_abstains_unsupported_sector"] = design("unsupported-sector-588", "Any", name)["status"]=="insufficient_evidence"
    ledger = [json.loads(x) for x in (OUTPUT / "api_ledger.jsonl").read_text().splitlines()]
    spend = sum(e["amount_usd"] for e in ledger)
    checks["within_authorised_2_usd_cap"] = spend <= 2.0
    report = {"verified_at": datetime.now(timezone.utc).isoformat(),
              "passed": all(checks.values()), "checks": {k: bool(v) for k,v in checks.items()},
              "scenarios": scenarios, "metric_coverage": COVERAGE,
              "reviewed_records_revalidated": quote_count,
              "invalid_evidence_records": bad_quotes,
              "implementation_sha256": {f.name: digest(f) for f in ROOT.glob('*.py')},
              "conservative_api_spend_usd": spend,
              "warning": "Technical integrity checks do not establish AI-label accuracy or independent ground truth."}
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2))
    (OUTPUT / "technical_summary.json").write_text(json.dumps({
        "verified": report["passed"], "review_coverage": comparison["review_coverage"],
        "model_results": comparison["results"], "student": comparison["student"],
        "scenarios": summaries, "api_spend_upper_estimate_usd": spend,
        "agentic_evaluation": json.loads((OUTPUT/'agentic/evaluations/course_agent_v1/summary.json').read_text()) if (OUTPUT/'agentic/evaluations/course_agent_v1/summary.json').exists() else {"status":"not_run"},
        "remaining": ["Full-population LLM review is not authorised or completed by the $2 pilot.",
                      "No independent new human gold standard; old labels and observed-demand definitions differ.",
                      "Agent usefulness ratings require actual reviewers; see outputs/agentic/evaluations/course_agent_v1/summary.json for live technical evaluation."]}, indent=2))
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "failed": [k for k,v in checks.items() if not v], "api_spend_upper_usd": spend}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
