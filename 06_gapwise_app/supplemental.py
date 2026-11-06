"""Cross-scenario metrics from the progress analysis plus auditable course decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import OUTPUT, load_jobs, load_courses
from taxonomy import maturity_level
from statsmodels.stats.proportion import proportion_confint

SKILLS = {
    "Cloud": r"\b(?:cloud|aws|azure|gcp)\b", "Python": r"\bpython\b", "SQL": r"\bsql\b",
    "Machine learning": r"machine learning|\bml\b", "Deep learning": r"deep learning|neural networks?",
    "NLP": r"natural language processing|\bnlp\b", "Computer vision": r"computer vision|image recognition",
    "Generative AI": r"generative ai|\bgenai\b", "LLM": r"large language model|\bllms?\b",
    "RAG": r"retrieval.augmented|\brag\b", "AI agents": r"\bai agents?\b|\bagentic\b",
    "MLOps": r"\bmlops\b|model serving|model monitoring", "Data engineering": r"data engineer|data pipelines?|\betl\b",
    "Responsible AI": r"responsible ai|ai ethics|ai governance|explainab",
}
DOMAINS = {
    "Finance": (r"Banking and Finance|Insurance|Accounting", r"Finance, Banking and Insurance AI"),
    "Engineering/manufacturing": (r"Engineering|Manufacturing", r"Manufactur|Industrial|Engineering AI"),
    "Marketing": (r"Marketing|Advertising", r"Marketing|Advertising"),
    "Logistics": (r"Logistics|Supply Chain", r"Logistics|Supply Chain"),
    "Healthcare": (r"Healthcare|Sciences|Pharmaceutical", r"Healthcare|Medical|Health AI"),
    "Education": (r"Education|Training", r"Education|Learning AI"),
    "HR": (r"Human Resources", r"Human Resources|HR AI"),
}
BUNDLES = {
    "AI_ML": ["Machine learning", "Deep learning", "NLP", "Computer vision"],
    "GenAI": ["Generative AI", "LLM", "RAG", "AI agents"],
    "Data": ["Python", "SQL", "Data engineering"],
    "Deployment_governance": ["Cloud", "MLOps", "Responsible AI"],
}


def build(scenario="expanded"):
    base = OUTPUT / "metrics" / scenario
    if not (base / "run_manifest.json").exists() or json.loads((base / "run_manifest.json").read_text())["status"] != "complete":
        raise ValueError("Complete the full metric scenario before supplemental analysis")
    jobs = load_jobs()
    cet, uni = load_courses()
    uni = uni[(uni.provider!="NUS") | uni.is_currently_offered.eq(True)]
    courses = pd.concat([cet.assign(ecosystem="CET"), uni.assign(ecosystem="University")], ignore_index=True)
    jl = pd.read_csv(OUTPUT / "job_labels.csv").set_index("record_id")
    cl = pd.read_csv(OUTPUT / "course_labels.csv").set_index("record_id")
    field = {"legacy": "legacy_ai", "strict": "strict_ai", "expanded": "expanded_ai", "student": "student_ai"}[scenario]
    jf = jobs.record_id.map(jl[field]).astype(bool)
    cf = courses.record_id.map(cl["expanded_ai" if scenario=="student" else field]).astype(bool)
    j, c = jobs[jf].copy(), courses[cf].copy()
    dest = base / "tables"
    low, high = proportion_confint(len(j), len(jobs), method="wilson")
    cet_ai = cet[cet.record_id.map(cl["expanded_ai" if scenario=="student" else field]).astype(bool)]
    provider_counts = cet_ai.groupby("trainingprovideralias").size().sort_values(ascending=False)
    shares = provider_counts / provider_counts.sum()
    experience = pd.to_numeric(j["Min Years Exp"], errors="coerce")
    pd.DataFrame([{"scenario": scenario, "jobs": len(jobs), "ai_jobs": len(j),
                   "ai_share": len(j)/len(jobs), "conditional_wilson_low": low,
                   "conditional_wilson_high": high, "cet_ai_courses": len(cet_ai),
                   "provider_hhi": float((shares**2).sum()),
                   "top5_provider_share": float(shares.head(5).sum()),
                   "top10_provider_share": float(shares.head(10).sum()),
                   "experience_known_n": int(experience.notna().sum()),
                   "experience_0_1_n": int(experience.le(1).sum()),
                   "experience_0_1_share_known": float(experience.dropna().le(1).mean()),
                   "caveat": "Rate interval conditional on labels; does not include classification or sampling-frame error."}]).to_csv(dest / "headline_metrics.csv", index=False)
    pd.DataFrame({"minimum_experience": experience}).assign(
        band=pd.cut(experience, [-.1, 1, 3, 5, 10, 60], labels=["0-1", "2-3", "4-5", "6-10", "11+"])
    ).groupby("band", observed=False).size().rename("jobs").to_csv(dest / "experience_barriers.csv")
    rows = []
    jskills, cskills = {}, {}
    for skill, pat in SKILLS.items():
        jm = j.review_text.str.contains(pat, case=False, regex=True)
        cm = c.review_text.str.contains(pat, case=False, regex=True)
        jskills[skill], cskills[skill] = jm, cm
        jp, cp = jm.mean(), cm.mean()
        rows.append({"skill": skill, "job_mentions": int(jm.sum()), "course_mentions": int(cm.sum()),
                     "job_denominator": len(j), "course_denominator": len(c),
                     "job_share": jp, "course_share": cp, "difference_pp": 100*(jp-cp),
                     "mention_ratio": jp/cp if cp else np.nan})
    pd.DataFrame(rows).to_csv(dest / "skill_mention_gaps.csv", index=False)
    js, cs = pd.DataFrame(jskills), pd.DataFrame(cskills)
    bundle_frames = []
    for name, frame in [("jobs", js), ("courses", cs)]:
        bundles = pd.DataFrame({bundle: frame[skills].any(axis=1) for bundle, skills in BUNDLES.items()})
        count = bundles.sum(axis=1)
        eligible = count.gt(0)
        out = count[eligible].value_counts().sort_index().rename_axis("bundle_count").reset_index(name="records")
        out["ecosystem"] = name
        out["denominator_with_any_bundle"] = int(eligible.sum())
        out["share"] = out.records/eligible.sum()
        bundle_frames.append(out)
    pd.concat(bundle_frames).to_csv(dest / "bundle_depth.csv", index=False)
    maturity = []
    for name, frame, labels in [("jobs", j, jl), ("courses", c, cl)]:
        levels = frame.review_text.map(maturity_level)
        known = frame.record_id.map(labels.maturity)
        levels.loc[known.gt(0)] = known[known.gt(0)].astype(int)
        m = levels.value_counts(normalize=False).sort_index().rename_axis("maturity").reset_index(name="records")
        m["ecosystem"], m["denominator"], m["share"] = name, len(frame), m.records/len(frame)
        maturity.append(m)
    pd.concat(maturity).to_csv(dest / "job_course_maturity.csv", index=False)
    domains = []
    for domain, (jpat, cpat) in DOMAINS.items():
        nj = int(j.Category.fillna("").str.contains(jpat, case=False, regex=True).sum())
        # Domain theme screening is a dictionary proxy, not an audited syllabus mapping.
        nc = int(c["AI Categories"].fillna("").str.contains(cpat, case=False, regex=True).sum())
        domains.append({"domain": domain, "unique_jobs": nj, "course_entries": nc,
                        "courses_per_100_jobs": 100*nc/nj if nj else np.nan,
                        "interpretation": "Catalogue-count/domain-tag proxy; not capacity or unmet enrolment demand"})
    pd.DataFrame(domains).to_csv(dest / "sector_course_ratios.csv", index=False)
    pd.DataFrame([
        {"scenario": scenario, "adoption_task": task, "jobs": int(jobs.record_id.map(jl.adoption_tasks.fillna("")).str.contains(task, regex=False).sum()),
         "observed_demand": False}
        for task in ["document_retrieval", "customer_support", "report_preparation", "software_assistance", "security_triage"]
    ]).to_csv(dest / "adoption_opportunities.csv", index=False)
    # Preserve all reviewer-generated task hypotheses and their evidence; the fixed
    # categories above cover only the deterministic baseline's standard taxonomy.
    adoption_rows = []
    for r in jl.itertuples():
        for opportunity in json.loads(r.adoption_evidence):
            adoption_rows.append({"record_id": r.Index, "task": opportunity["task"],
                                  "quote": opportunity["quote"], "assumption": opportunity["assumption"],
                                  "source": r.screening_source, "observed_demand": False})
    pd.DataFrame(adoption_rows).to_csv(dest / "adoption_task_evidence.csv", index=False)

    # Data-derived candidate scores replace manual ordinal scores.
    sec = pd.read_csv(dest / "sec.csv", index_col=0)
    candidates = [
        ("GenAI workflows for finance professionals", "Banking and Finance", ["LLM", "RAG", "Responsible AI"]),
        ("AI integration for software teams", "Information Technology", ["LLM", "RAG", "AI agents"]),
        ("Production AI for ML practitioners", "Information Technology", ["MLOps", "Cloud", "Python"]),
        ("Applied AI for engineering teams", "Engineering", ["Machine learning", "Python", "MLOps"]),
        ("Governed AI for consultants", "Consulting", ["Responsible AI", "Generative AI", "LLM"]),
        ("AI-enabled marketing workflows", "Marketing / Public Relations", ["Generative AI", "LLM", "Responsible AI"]),
    ]
    scores = []
    retrieved_evidence = {}
    vectorizer = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 2))
    course_vectors = vectorizer.fit_transform(c.review_text)
    for concept, sector, skills in candidates:
        row = sec[sec.sector.eq(sector)]
        mask = j.Category.fillna("").str.split("; ").map(lambda x: sector in x)
        count = int(mask.sum())
        if row.empty or count < 25:
            continue
        demand = js.loc[mask, skills].mean()
        supply = cs[skills].mean()
        gap = float((demand-supply).clip(lower=0).mean())
        # Same declared scalar rule for every candidate; no hand-picked 1-5 values.
        component = row.iloc[0]
        final = .8*float(component.Opportunity) + .2*gap
        example_mask = mask & j.record_id.map(jl.expanded_ai).astype(bool)
        if sector == 'Banking and Finance':
            example_mask &= j.Title.str.contains(r'financ|bank|treasur|compliance|accountant|risk', case=False, regex=True)
        examples = j.loc[example_mask, ["record_id", "Title", "Job URL"]].copy()
        examples["reviewed_by_llm"] = examples.record_id.map(jl.reviewed_by_llm)
        examples = examples.sort_values("reviewed_by_llm", ascending=False).head(3)
        job_examples = []
        for r in examples.itertuples(index=False):
            label = jl.loc[r.record_id]
            quote_field = label.teacher_evidence if pd.notna(label.teacher_evidence) else label.rule_evidence
            job_examples.append({"record_id": r.record_id, "title": r.Title,
                                 "url": r[2] if pd.notna(r[2]) else None,
                                 "source": label.screening_source,
                                 "quotes": json.loads(quote_field)})
        query = concept + " " + " ".join(skills)
        if sector == 'Banking and Finance':
            query = 'finance financial banking generative artificial intelligence retrieval augmented generation policy documents governance'
        similarities = cosine_similarity(vectorizer.transform([query]), course_vectors)[0]
        course_examples = []
        for k in np.argsort(similarities)[-3:][::-1]:
            course = c.iloc[k]
            title = course.get("coursetitle") if course.ecosystem=="CET" else course.get("course_name")
            course_examples.append({"record_id": course.record_id, "title": str(title),
                                    "ecosystem": course.ecosystem, "lexical_similarity": float(similarities[k]),
                                    "excerpt": course.review_text[:500],
                                    "interpretation": "Retrieved competitor candidate; syllabus equivalence is not established."})
        retrieved_evidence[concept] = {"jobs": job_examples, "competing_courses": course_examples}
        scores.append({"concept": concept, "sector": sector, "ai_jobs": count,
                       "sector_opportunity": float(component.Opportunity), "skill_mention_gap": gap,
                       "score": final, "skills": "; ".join(skills),
                       "score_rule": "0.8*sector_opportunity + 0.2*mean_positive_skill_share_gap",
                       "qualification": "Decision support; weights are judgemental and skill mentions are proxies"})
    result = pd.DataFrame(scores)
    if not result.empty:
        result = result.sort_values("score", ascending=False)
    result.to_csv(dest / "candidate_scores_v4.csv", index=False)
    (base / "candidate_evidence.json").write_text(json.dumps(retrieved_evidence, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=["legacy", "strict", "expanded"])
    a = p.parse_args()
    for scenario in a.scenarios:
        print(scenario, build(scenario).to_string(index=False))
