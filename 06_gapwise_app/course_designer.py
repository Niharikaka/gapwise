"""Evidence retrieval and deterministic course-outline prototype. No API spend.

This is a functioning evidence-based workflow, not an autonomous LLM agent.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from data import OUTPUT


def design(sector, audience, scenario="expanded"):
    base = OUTPUT / "metrics" / scenario
    candidates = pd.read_csv(base / "tables/candidate_scores_v4.csv")
    match = candidates[candidates.sector.str.contains(sector, case=False, regex=False)]
    if match.empty:
        return {"status": "insufficient_evidence", "sector": sector,
                "reason": "No eligible candidate with sufficient analysed demand for this sector.",
                "available_sectors": candidates.sector.unique().tolist()}
    row = match.sort_values("score", ascending=False).iloc[0]
    evidence_path = base / "candidate_evidence.json"
    retrieval = json.loads(evidence_path.read_text())[row.concept]
    skills = row.skills.split("; ")
    modules = [{"module": i+1, "skill": skill,
                "learning_outcome": f"Apply {skill} to a documented {row.sector} use case and explain its limitations.",
                "assessment": "Submit a reproducible task artefact with evidence checks."}
               for i, skill in enumerate(skills)]
    return {"status": "draft_requires_provider_review", "scenario": scenario, "sector": row.sector,
            "audience": audience, "concept": row.concept, "ai_job_count": int(row.ai_jobs),
            "score": float(row.score), "modules": modules,
            "supporting_jobs": retrieval["jobs"], "competing_courses": retrieval["competing_courses"],
            "prerequisites": "Provider must assess learners' domain and technical prerequisites.",
            "duration": "To be estimated after assessment and syllabus review; not inferred from job counts.",
            "evidence": [str(base / "tables/candidate_scores_v4.csv"), str(base / "tables/sec.csv"),
                         str(base / "tables/skill_mention_gaps.csv"), str(base / "run_manifest.json"), str(evidence_path)],
            "limitations": ["Candidate scores are decision proxies, not predicted course sales.",
                            "Pilot LLM coverage is partial; remaining records are identified screening/model outputs.",
                            "The template draft is deterministic, not an LLM-generated or provider-validated syllabus."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sector", required=True)
    p.add_argument("--audience", default="Working professionals in the selected sector")
    p.add_argument("--scenario", default="expanded")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = design(a.sector, a.audience, a.scenario)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
