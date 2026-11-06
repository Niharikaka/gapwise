"""Data layer for the Gapwise prototype (gapwise_app.py).

Reads existing analytical outputs and the existing provider engine; it computes no new
decision rules. Verdicts come from provider_project.research.analyse (the six rules in
provider_project/policy.json). Course-integration suggestions come from course_infusion.py.
`python3 gapwise_data.py build` precomputes the two heavier tables into outputs/gapwise/.
"""
from __future__ import annotations
from functools import lru_cache
import json
import re
import sys
import pandas as pd
from data import OUTPUT
from supplemental import SKILLS
from course_infusion import TASKS, SECTORS as INFUSION_SECTORS, CROSS, sector_of, course_frame, merge_variants

SCENARIO = "strict"
METRICS = OUTPUT / "metrics" / SCENARIO / "tables"
INFUSION = OUTPUT / "infusion"
GAPWISE = OUTPUT / "gapwise"
LEVELS = {1: "Aware", 2: "Use", 3: "Integrate", 4: "Build", 5: "Operate"}
# MyCareersFuture sector (used by the verdict engine) -> course_infusion industry (used by
# course paths). Sectors without an industry fall back to cross-industry courses.
INDUSTRY = {"Banking and Finance": "Finance and accounting", "Accounting / Auditing / Taxation": "Finance and accounting",
            "Insurance": "Finance and accounting", "Human Resources": "Human resources",
            "Logistics / Supply Chain": "Logistics and supply chain", "Building and Construction": "Building and construction",
            "Customer Service": "Customer service", "Marketing / Public Relations": "Marketing and sales",
            "Sales / Retail": "Marketing and sales", "Advertising / Media": "Marketing and sales",
            "Healthcare / Pharmaceutical": "Healthcare", "Medical / Therapy Services": "Healthcare",
            "Admin / Secretarial": "Admin and office"}
FOCUS_SKILLS = ["LLM", "RAG", "AI agents", "Responsible AI"]
MIN_RADAR_JOBS = 800
# Declared: lexical similarity at which a concept counts as matching the user's description.
MIN_BRIEF_MATCH = 0.05


def industry(sector: str) -> str:
    return INDUSTRY.get(sector, CROSS)


def ladder() -> pd.DataFrame:
    """Share of AI jobs and AI courses at each AI Depth Ladder level (strict scenario)."""
    m = pd.read_csv(METRICS / "job_course_maturity.csv")
    m = m[m.maturity.between(1, 5)].pivot(index="maturity", columns="ecosystem", values="share").reset_index()
    m["name"] = m.maturity.map(LEVELS)
    m["gap"] = m.jobs - m.courses
    return m.rename(columns={"jobs": "job_share", "courses": "course_share"})


def headline() -> dict:
    s = json.loads((OUTPUT / "metrics" / SCENARIO / "summary.json").read_text())
    lad = ladder().set_index("maturity")
    return {"jobs": s["jobs"], "ai_jobs": s["ai_jobs"], "ai_rate": s["ai_rate"],
            "ai_courses": s["cet_ai_courses"] + s["university_ai_courses"],
            "courses_at_l1": float(lad.loc[1, "course_share"])}


def sectors() -> pd.DataFrame:
    """Sectors the verdict engine can analyse, largest AI demand first."""
    s = pd.read_csv(METRICS / "sec.csv")
    s = s[~s.sector.eq("Others")].sort_values("ai_jobs", ascending=False)
    return s[["sector", "jobs", "ai_jobs", "ai_rate", "Opportunity"]].reset_index(drop=True)


def opportunities() -> pd.DataFrame:
    """Hiring-now ideas: the existing ranked candidate table, one per sector."""
    c = pd.read_csv(METRICS / "candidate_scores_v4.csv").sort_values("score", ascending=False)
    return c.drop_duplicates("sector").reset_index(drop=True)


def radar() -> pd.DataFrame:
    p = GAPWISE / "radar.csv"
    if not p.exists():
        raise FileNotFoundError("Run `python3 gapwise_data.py build` first")
    r = pd.read_csv(p)
    r["zone"] = [zone(a, l) for a, l in zip(r.ai_share, r.latent_share)]
    return r


def zone(ai_share: float, latent_share: float) -> str:
    """Declared display cut-offs: 5% asking for AI, 35% with AI-ready tasks."""
    if latent_share >= 0.35:
        return "Build ahead" if ai_share < 0.05 else "Hiring now"
    return "Quiet" if ai_share < 0.05 else "Specialist"


def course_skills() -> pd.DataFrame:
    p = GAPWISE / "course_skills.csv"
    if not p.exists():
        raise FileNotFoundError("Run `python3 gapwise_data.py build` first")
    return pd.read_csv(p)


def new_path(sector: str) -> dict:
    """White space among AI courses in the sector's industry: how many mention each focus skill."""
    c = course_skills()
    c = c[c.industry.eq(industry(sector))]
    counts = {s: int(c[s].sum()) for s in FOCUS_SKILLS}
    return {"industry": industry(sector), "ai_courses": len(c), "counts": counts,
            "all_focus": int(c[FOCUS_SKILLS].all(axis=1).sum()),
            "open": [s for s, n in sorted(counts.items(), key=lambda x: x[1]) if n <= max(2, 0.25*max(counts.values() or [0]))]}


def adapt_options(sector: str, limit: int = 5) -> pd.DataFrame:
    """AI courses in the industry that miss at least one focus skill, most attended first."""
    c = course_skills()
    c = c[c.industry.eq(industry(sector))].copy()
    c["has"] = c[FOCUS_SKILLS].apply(lambda r: [s for s in FOCUS_SKILLS if r[s]], axis=1)
    c["missing"] = c[FOCUS_SKILLS].apply(lambda r: [s for s in FOCUS_SKILLS if not r[s]], axis=1)
    c = c[c.has.str.len().gt(0) & c.missing.str.len().gt(0)]
    return c.sort_values(["attendance", "title"], ascending=[False, True], na_position="last").head(limit).reset_index(drop=True)


def infuse_options(sector: str, query: str = "", limit: int = 6) -> pd.DataFrame:
    """Non-AI courses in the industry with their AI suggestion; search narrows by title."""
    s = pd.read_csv(INFUSION / "integration_suggestions.csv")
    s = s[s.sector.eq(industry(sector))]
    if query.strip():
        s = s[s.title.str.contains(query.strip(), case=False, regex=False, na=False)]
    return s.sort_values(["attendance", "title"], ascending=[False, True], na_position="last").head(limit).reset_index(drop=True)


def task_demand_map(sector: str) -> dict:
    """task -> (share of the industry's job ads describing it, share of those asking for AI)."""
    t = pd.read_csv(INFUSION / "task_demand.csv")
    t = t[t.sector.eq(industry(sector))]
    return {r.task: (float(r.task_share), float(r.ai_share_of_task_jobs) if r.ai_share_of_task_jobs == r.ai_share_of_task_jobs else 0.0) for r in t.itertuples()}


def similar_courses(sector: str, skills: list[str], limit: int = 5) -> pd.DataFrame:
    """AI courses in the industry that mention any of the given focus skills (for comparison only)."""
    c = course_skills()
    c = c[c.industry.eq(industry(sector))]
    cols = [s for s in skills if s in c.columns]
    if cols:
        c = c[c[cols].any(axis=1)]
    return c.sort_values(["attendance", "title"], ascending=[False, True], na_position="last").head(limit)


def infusion_totals(sector: str) -> dict:
    return json.loads((INFUSION / "summary.json").read_text())["sectors"][industry(sector)]


def horizons(sector: str) -> dict:
    """Now / Next / Later evidence. Later is AI-ready tasks: a hypothesis, never demand."""
    r = radar().set_index("sector")
    row = r.loc[sector] if sector in r.index else None
    t = pd.read_csv(INFUSION / "task_demand.csv")
    t = t[t.sector.eq(industry(sector))].sort_values("task_share", ascending=False)
    # Industries without their own task screen would borrow all-job tasks (e.g. customer
    # enquiries for a software course), so they get no forward-looking task.
    ahead = t[t.ai_share_of_task_jobs.lt(0.05)].head(2) if industry(sector) != CROSS else t.iloc[:0]
    it = r.loc["Information Technology"] if "Information Technology" in r.index else None
    return {"ai_share": float(row.ai_share) if row is not None else None,
            "latent_share": float(row.latent_share) if row is not None else None,
            "it_ai_share": float(it.ai_share) if it is not None else None,
            "ahead_tasks": [{"task": x.task, "label": TASKS[x.task]["label"], "task_share": float(x.task_share),
                             "ai_share": float(x.ai_share_of_task_jobs), "addon": TASKS[x.task]["addon"]} for x in ahead.itertuples()]}


def verdict(sector: str, audience: str, need: str, level_min: int, level_max: int, hours: int) -> dict:
    """Run the existing provider engine and reduce its bundle to what the screens show."""
    from provider_project.brief import ProviderBrief
    from provider_project.research import analyse
    from provider_decisions import rule_explanation
    bundle = analyse(ProviderBrief(sector, audience, need, level_min, level_max, hours, scenario=SCENARIO))
    cands = bundle["candidates"]
    engine = next((c for c in cands if c["candidate_id"] == bundle.get("recommended_candidate_id") and c["eligible"]), None)
    # The user described a specific course, so judge the concept closest to their words when it
    # is a clear match; otherwise fall back to the engine's highest-scoring eligible concept.
    closest = max(cands, key=lambda c: c["brief_similarity"]) if cands else None
    matched = closest if closest and closest["brief_similarity"] >= MIN_BRIEF_MATCH else None
    top = matched or engine or (cands[0] if cands else None)
    rec = top if top and top["eligible"] else None
    alternative = engine["concept"].split(" · ")[0] if engine and top and engine["candidate_id"] != top["candidate_id"] else ""
    checks = [{"label": r["label"], "passed": bool(r["passed"]), "why": rule_explanation(r)} for r in (top["six_rules"] if top else [])]
    return {"go": rec is not None, "concept": top["concept"].split(" · ")[0] if top else "",
            "key": top["key"] if top else "", "skills": top["skills"] if top else [], "levels": top["overlap_levels"] if top else [],
            "sector_ai_jobs": bundle.get("sector_ai_jobs"), "sector_jobs": bundle.get("sector_all_jobs"),
            "courses_per_100": bundle.get("sector_courses_per_100_ai_jobs"), "checks": checks,
            "job_examples": [{"title": j["title"], "quote": j["quote"]} for j in (top["job_examples"][:3] if top else [])],
            "skill_evidence": top["skill_evidence"] if top else [], "interpretation": bundle.get("interpretation", ""),
            "matched_description": matched is not None, "alternative": alternative, "alternative_key": engine["key"] if alternative else ""}


def ideas(sector: str, limit: int = 3) -> list[dict]:
    """Course ideas for an industry, ranked by the existing engine under a neutral brief
    (all depths, generic learners) so no idea is favoured by wording the user never gave."""
    from provider_project.brief import ProviderBrief
    from provider_project.research import analyse, CATALOGUE
    bundle = analyse(ProviderBrief(sector, "Working professionals", "Apply AI tools to everyday work tasks in this industry", 1, 5, 24, scenario=SCENARIO))
    intent = {c["key"]: c["intent"] for c in CATALOGUE}
    # Eligibility is not shown here: the audience and fit rules need the user's real brief.
    ranked = sorted(bundle["candidates"], key=lambda c: c["score"], reverse=True)[:limit]
    return [{"concept": c["concept"].split(" · ")[0], "key": c["key"], "skills": c["skills"], "levels": c["levels"],
             "jobs": c["candidate_skill_jobs"], "intent": intent[c["key"]]} for c in ranked]


# Declared cut-offs for comparing a proposed objective with the nearest published outcome
# (TF-IDF cosine). Lexical overlap only: it does not show equal depth or assessment.
TAUGHT_SIM, PARTLY_SIM, MIN_COURSE_SIM = 0.35, 0.15, 0.05
MAX_POINT = 300
AI_TERMS = re.compile(r"\bai\b|artificial intelligence|machine learning|generative|chatgpt|copilot|language model|automat", re.I)


@lru_cache(maxsize=1)
def catalogue():
    """Every course (variants merged) with its published outcomes and a text index for retrieval."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from data import load_courses
    cet, uni = load_courses()
    labels = pd.read_csv(OUTPUT / "course_labels.csv")
    c = merge_variants(course_frame(cet, uni, labels)).reset_index(drop=True)
    extra = cet.set_index("record_id")[["full_course_fee", "courseratings_value"]]
    c = c.join(extra, on="record_id")
    text = (c.title.fillna("") + " " + c.outcomes.fillna("") + " " + c.description.fillna("")).tolist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=60000, sublinear_tf=True)
    return c, vec, vec.fit_transform(text)


# Coverage uses broader synonyms than the job-side skill dictionary: courses name tools
# ("ChatGPT", "knowledge base") more often than the technical term.
COVERAGE = {"LLM": r"large language model|\bllms?\b|chatgpt|\bgpt\b|copilot|chatbot|language model|prompt",
            "RAG": r"retrieval|\brag\b|knowledge base|vector (?:database|store)|embedding|document (?:search|q&a|question)",
            "AI agents": r"\bagents?\b|agentic|ai assistant|autonomous|workflow automation",
            "Responsible AI": r"responsible ai|ethic|governance|\bbias|privacy|explainab|trustworthy",
            "Generative AI": r"generative|\bgen ?ai\b|chatgpt|copilot|midjourney|dall-?e"}
LEAD_IN = re.compile(r"^.*?(?:able to|will learn|participants will|learners will|you will)\s*:?\s*", re.I)
SPLIT = re.compile(r"\s*(?:•|▪|●|·|;|\n|\s-\s|(?<![\w.])\d{1,2}[.)]\s|\bLO\s*\d+\s*[:.\-]?|\bLU\s*\d+\s*[:.\-]?)\s*")


def outcome_points(outcomes: str, limit: int = 8) -> list[str]:
    """Published outcome text split into readable points (lead-in and numbering removed)."""
    text = LEAD_IN.sub("", outcomes or "", count=1) if len(outcomes or "") > 80 else (outcomes or "")
    pts = [p.strip(" .,:;-–").rstrip(" 0123456789") for p in SPLIT.split(text)]
    pts = [p[0].upper() + p[1:] for p in pts if len(p) >= 20]
    return pts[:limit]


def skill_status(row, skills: list[str]) -> dict:
    """Per skill: 'outcomes' if named in published outcomes, 'description' if only described, else 'missing'."""
    out = {}
    for s in skills:
        pattern = COVERAGE.get(s) or SKILLS.get(s)
        rx = re.compile(pattern, re.I) if pattern else None
        if rx and rx.search(row.outcomes or ""):
            out[s] = "outcomes"
        elif rx and rx.search(f"{row.title} {row.description or ''}"):
            out[s] = "description"
        else:
            out[s] = "missing"
    return out


def advice(row, status: dict, mode: str) -> tuple[str, str]:
    """(label, sentence) for one existing course, from declared rules over its published text."""
    from gapwise_objectives import PLAIN_SKILL
    plain = lambda xs: [PLAIN_SKILL.get(x, x) for x in xs]
    taught = plain([s for s, v in status.items() if v == "outcomes"])
    described = plain([s for s, v in status.items() if v == "description"])
    missing = plain([s for s, v in status.items() if v == "missing"])
    if mode == "existing" and not row.has_ai:
        return "Add AI", "A similar course with no AI yet, like yours. Adding AI would set yours apart."
    if not row.outcomes:
        return "Check syllabus", "No learning outcomes are published, so coverage is unknown."
    if status and not missing and not described:
        return "Differentiate", "Already teaches all of these skills in its outcomes. Stand out on depth, industry focus or assessment."
    if taught or described:
        have = ", ".join(taught + described)
        return "Adapt or partner", f"Covers {have} but not {', '.join(missing) or 'every skill in its outcomes'}. A close course to upgrade, or to beat on what it misses."
    return "Related", "Similar topic, but it does not teach these skills."


def course_facts(description: str, skills: list[str], mode: str = "new", limit: int = 5, sector: str = "") -> list[dict]:
    """Closest real courses to the user's description, with their published outcomes, skill
    coverage and advice. New ideas compare with AI courses; existing courses with all courses.
    Courses from the same industry are compared first; others only if too few are close."""
    from sklearn.metrics.pairwise import linear_kernel
    import numpy as np
    c, vec, M = catalogue()
    query = description + " " + " ".join(skills) if mode == "new" else description
    scores = linear_kernel(vec.transform([query]), M).ravel()
    base = c.has_ai.to_numpy() if mode == "new" else np.ones(len(c), bool)
    same = base & c.sector.eq(industry(sector)).to_numpy() if sector and industry(sector) != CROSS else base
    pool = np.where(same)[0]
    if (scores[pool] >= MIN_COURSE_SIM).sum() < 3:
        pool = np.where(base)[0]
    order = pool[np.argsort(-scores[pool], kind="stable")][:limit * 4]
    facts, seen = [], set()
    for i in order:
        if scores[i] < MIN_COURSE_SIM or len(facts) == limit:
            break
        r = c.iloc[i]
        # Variant merging keys on exact titles; punctuation differences still repeat a course.
        key = (re.sub(r"[^a-z0-9]", "", str(r.base_title).lower()), str(r.provider_name).lower())
        if key in seen:
            continue
        seen.add(key)
        status = skill_status(r, skills)
        label, why = advice(r, status, mode)
        facts.append({"record_id": r.record_id, "title": r.base_title, "provider": r.provider_name, "source": r.source,
                      "hours": r.hours, "fee": r.full_course_fee, "rating": r.courseratings_value, "attendance": r.attendance,
                      "has_ai": bool(r.has_ai), "industry": r.sector, "similarity": float(scores[i]),
                      "outcomes": outcome_points(r.outcomes), "has_outcomes": bool(r.outcomes), "skills": status,
                      "advice": label, "why": why})
    return facts


def objective_overlap(objectives: list[str], facts: list[dict], anchors: list[list[str]] | None = None) -> list[dict]:
    """For each proposed objective, the nearest published outcome among the compared courses.
    An outcome that names every key idea of the objective counts as partly covering it, even
    when the wording differs; text similarity alone decides 'Already taught'."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    # Only courses that use AI can already teach an AI objective; others teach the bare task.
    units = [(f["title"], u) for f in facts if f["has_ai"] for u in f["outcomes"]]
    if not units:
        return [{"status": "Unknown", "course": "", "outcome": "", "similarity": None} for _ in objectives]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit([u for _, u in units] + objectives)
    sim = linear_kernel(vec.transform(objectives), vec.transform([u for _, u in units]))
    out = []
    for n, row in enumerate(sim):
        j = int(row.argmax())
        s = float(row[j])
        status = "Already taught" if s >= TAUGHT_SIM else ("Partly covered" if s >= PARTLY_SIM else "New")
        ideas = anchors[n] if anchors and n < len(anchors) else []
        if status == "New" and ideas:
            # One short point that mentions AI and every key idea; long paragraphs match anything.
            hit = next((k for k, (_, u) in enumerate(units) if len(u) <= MAX_POINT and AI_TERMS.search(u)
                        and all(re.search(i, u.lower()) for i in ideas)), None)
            if hit is not None:
                status, j = "Partly covered", hit
        out.append({"status": status, "course": units[j][0] if status != "New" else "", "outcome": units[j][1] if status != "New" else "", "similarity": s})
    return out


def route(facts: list[dict], mode: str) -> tuple[str, str]:
    """Overall recommendation from the per-course advice (declared rule)."""
    labels = [f["advice"] for f in facts]
    if mode == "existing":
        n = sum(not f["has_ai"] for f in facts)
        return ("Add AI to your course", f"{n} of the {len(facts)} most similar courses have no AI yet, so adding it would set yours apart."
                if n else "Similar courses already include AI. Add it to keep up, and focus on the tasks your learners do.")
    if "Differentiate" in labels:
        return "Differentiate", "At least one existing course already teaches these skills. Make yours deeper, more industry-specific or better assessed."
    if "Adapt or partner" in labels:
        return "Adapt or build", "Close courses exist but each misses some skills. Upgrade or partner with one, or build a course that covers the gap."
    return "Build new", "Nothing close teaches these skills, so there is room for a new course."


def build():
    """Precompute the radar and per-course skill flags (reads the full job and course files)."""
    from data import load_jobs, load_courses
    GAPWISE.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs()[["record_id", "Category"]]
    lab = pd.read_csv(OUTPUT / "job_labels.csv", usecols=["record_id", "strict_ai", "adoption_tasks"])
    j = jobs.merge(lab, on="record_id", how="left")
    j["strict_ai"] = j.strict_ai.fillna(False).astype(bool)
    j["latent"] = ~j.strict_ai & j.adoption_tasks.fillna("").astype(str).str.len().gt(2)
    j["sector"] = j.Category.fillna("").str.split("; ")
    j = j.explode("sector")
    r = j.groupby("sector").agg(jobs=("record_id", "nunique"), ai_share=("strict_ai", "mean"), latent_share=("latent", "mean")).reset_index()
    r = r[r.jobs.ge(MIN_RADAR_JOBS) & ~r.sector.isin(["", "Others"])]
    r.to_csv(GAPWISE / "radar.csv", index=False)

    cet, uni = load_courses()
    labels = pd.read_csv(OUTPUT / "course_labels.csv")
    c = merge_variants(course_frame(cet, uni, labels))
    strict = set(labels.record_id[labels.strict_ai.astype(bool)])
    c = c[c.variant_ids.str.split("; ").map(lambda ids: any(i in strict for i in ids))].copy()
    text = (c.title.fillna("") + " " + c.outcomes.fillna("") + " " + c.description.fillna("")).str.lower()
    for s in FOCUS_SKILLS:
        c[s] = text.str.contains(SKILLS[s], regex=True)
    c["industry"] = c.title.map(sector_of)
    c[["record_id", "title", "base_title", "provider_name", "source", "hours", "attendance", "industry", *FOCUS_SKILLS]].to_csv(GAPWISE / "course_skills.csv", index=False)
    print(f"radar sectors {len(r)}, AI courses {len(c)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        print(__doc__)
