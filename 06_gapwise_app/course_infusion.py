"""AI-integration suggestions for every existing non-AI course, in three tiers.

L1 baseline: every course gets one. It is a policy choice, not a demand claim: job ads
under-report AI (the 400-ad benchmark put recall near 25%), so the absence of AI in ads is
not evidence that a field does not need AI literacy.
L2 task upgrade: the course's published text teaches a workplace task AI is changing, and
sector job ads commonly describe that task.
L3 build: only hiring-backed when enough AI-asking ads for the task sit at L3 or above;
otherwise it is labelled build ahead (anticipated), never counted as demand.

Task matches are transparent lexical evidence with source quotations, not audited syllabi.
Add-on wording comes from declared templates per task and sector; the skills and level
evidence beside them come from job ads.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from data import OUTPUT, load_jobs, load_courses
from dashboard_data import clean, usable_outcomes
from supplemental import SKILLS

# Each task: the same pattern is used on course text and job ads, so both sides are
# measured alike. level is the AI Depth Ladder rung the upgrade template targets.
TASKS = {
    "report_writing": dict(label="Writing reports and business documents",
        pattern=r"\b(?:writ\w*|prepar\w*|draft\w*)\b.{0,40}\b(?:reports?|proposals?|memos?|minutes|documentation)\b|business writing|report writing",
        addon="Drafting reports with AI, then fact-checking them", level="L2", hours=3),
    "budgeting_forecasting": dict(label="Budgeting and forecasting",
        pattern=r"\bbudget\w*|\bforecast\w*|variance analysis|cash ?flow|financial planning",
        addon="AI-assisted forecasting and variance commentary", level="L2", hours=4),
    "financial_statements": dict(label="Reading financial statements",
        pattern=r"financial statements?|balance sheets?|income statements?|ratio analysis|financial reporting",
        addon="Analysing statements with AI and checking its answers", level="L2", hours=3),
    "document_review": dict(label="Reviewing contracts, policies and regulations",
        # Generic "regulatory requirements" mostly means following rules, not reviewing documents.
        pattern=r"contract (?:review|management|drafting)|due diligence|policy documents?|regulatory (?:frameworks?|filings?|submissions?|reporting)|interpret\w* (?:legislation|regulations?|standards?|contracts?)",
        addon="Answering questions from approved documents (RAG)", level="L3", hours=6),
    "audit_reconciliation": dict(label="Audit, controls and reconciliation",
        # Bare "audit" catches brand, energy and log audits.
        pattern=r"(?:financial|internal|external|statutory) audit\w*|audit(?:ing)? (?:and|&) assurance|audit (?:process|procedures|evidence|plans?|findings|engagements?)|\bauditors?\b|internal controls?|reconcil\w*|fraud detection",
        addon="AI anomaly checks and reconciliation with human sign-off", level="L3", hours=4),
    "data_analysis": dict(label="Analysing data in spreadsheets and dashboards",
        # Case-sensitive Excel: lower-case "excel" is usually the verb.
        pattern=r"data analy\w*|analy[sz]e data|(?-i:\bExcel\b)|spreadsheets?|dashboards?|pivot tables?",
        addon="An AI copilot for spreadsheet analysis", level="L2", hours=3),
    "customer_communication": dict(label="Handling customer enquiries and complaints",
        pattern=r"customer (?:service|enquir\w*|complaints?|feedback)|handl\w* (?:complaints|enquiries)|client communication",
        addon="Working with AI assistants and supervising chat agents", level="L3", hours=4),
    "research_summarising": dict(label="Researching and summarising information",
        pattern=r"market research|literature review|desk research|summari[sz]\w*|synthesi[sz]e information",
        addon="AI research and summaries with source citations", level="L2", hours=3),
    "content_marketing": dict(label="Creating marketing content",
        pattern=r"content (?:creation|writing|marketing)|copywriting|social media (?:content|posts?|marketing)|marketing campaigns?",
        addon="Generative AI content with brand and accuracy checks", level="L2", hours=4),
    "recruitment": dict(label="Recruiting and screening candidates",
        # Bare "recruit"/"job description" is recruitment-agency boilerplate in most job ads.
        pattern=r"talent acquisition|recruitment (?:process|cycle|strategy|campaigns?|activities)|(?:end-to-end|full[- ]cycle) recruitment|screen\w* (?:candidates|applicants|resumes|cvs)|sourc\w* (?:candidates|talent)|interview\w* candidates",
        addon="AI-assisted screening with fairness checks", level="L2", hours=3),
    "tender_procurement": dict(label="Tenders, procurement and vendor evaluation",
        pattern=r"\btenders?\b|tendering|procurement|purchase orders?|vendor (?:evaluation|selection|management)|quotations?",
        addon="AI review of tender documents and specifications", level="L3", hours=4),
    "planning_scheduling": dict(label="Planning, scheduling and inventory",
        # Bare "scheduled" is usually timetabling ("scheduled practicum").
        pattern=r"\bscheduling\b|(?:project|production|work|shift|shoot|maintenance) schedul\w*|roster\w*|demand planning|inventory (?:management|control|planning)|route planning|capacity planning",
        addon="AI-assisted planning and optimisation", level="L2", hours=4),
    "inspection_quality": dict(label="Inspection and quality checks",
        pattern=r"site inspections?|quality (?:control|inspection|assurance)|defect (?:detection|inspection)|visual inspection",
        addon="Using AI image tools for inspection, and their limits", level="L2", hours=3),
}
# Course title patterns tag the course's industry; MyCareersFuture Category patterns pick
# the jobs it is compared with. Untagged courses are compared with all job ads.
SECTORS = {
    "Finance and accounting": (r"financ|accounting|accountan|(?:internal|external|financial) audit|audit(?:ing)? (?:and|&) assurance|\btax(?:ation)?\b|\bbank|fintech|treasury|payroll|insurance",
                               r"Banking and Finance|Accounting|Insurance"),
    "Human resources": (r"human resource|\bhr\b|recruit|talent (?:management|acquisition|development)|payroll", r"Human Resources"),
    "Logistics and supply chain": (r"logistic|supply chain|warehous|procurement|inventory|freight|shipping",
                                   r"Logistics|Supply Chain|Purchasing"),
    # Bare "building" would catch team-building courses.
    "Building and construction": (r"construction|\bbim\b|quantity survey|building (?:works|services|information|control|maintenance)\b|facilit(?:y|ies) management|\btender",
                                  r"Building and Construction|Architecture|Property"),
    "Customer service": (r"customer service|customer experience|service excellence|call cent|contact cent",
                         r"Customer Service"),
    "Marketing and sales": (r"marketing|\bsales\b|branding|advertis|social media|e-?commerce",
                            r"Marketing|Advertising|Sales|Public Relations"),
    # Bare "health" would catch workplace safety and health courses.
    "Healthcare": (r"healthcare|nursing|patient|clinical|medical|pharma|caregiv|elder ?care|radiolog|dental|physiother|hospital", r"Healthcare|Medical|Pharmaceutical"),
    # Bare "administration" catches MBAs and IT system admins.
    "Admin and office": (r"secretar|office admin\w*|administrative (?:support|assistant|functions|skills)|executive assistant|office management",
                         r"Admin|Secretarial"),
}
CROSS = "Cross-industry"
# L1 baseline wording by industry, used when no task is matched; with a task it is tailored.
BASELINE = {
    "Finance and accounting": "Using AI safely with financial data: drafting, checking figures, client confidentiality",
    "Human resources": "Using AI fairly with people data: drafting, the limits of screening, privacy",
    "Logistics and supply chain": "Using AI in operations: summaries, planning help, checking its output",
    "Building and construction": "Using AI on projects and sites: documents, photos, safety limits",
    "Customer service": "Using AI assistants with customers: drafting replies, when to hand over to a person",
    "Marketing and sales": "Using generative AI for content: brand, accuracy and copyright checks",
    "Healthcare": "Using AI in care settings: documentation help, patient privacy, clinical limits",
    "Admin and office": "Using AI for office work: emails, minutes, scheduling, data privacy",
    CROSS: "Using AI tools in this field: what they do well, checking their output, protecting data",
}
BASELINE_HOURS = 2
# Hands-on, licensing or regulated courses still get a baseline, flagged as optional:
# an AI module may not fit their assessed content or regulator requirements.
HANDS_ON = re.compile(r"food safety|food hygiene|first aid|\bcpr\b|\baed\b|workplace safety|\bwsh\b|safety (?:and|&) health|forklift|"
                      r"crane|scaffold|confined space|work(?:ing)? at height|security officer|threat observation|lifeguard|"
                      r"driving|\blicen[cs]e\b|fire safety", re.I)
# Declared decision rules (judgemental, like agentic/policy.json): not validated thresholds.
RULES = {"min_task_share_for_l2": 0.10, "min_ai_task_ads_for_l3": 25, "min_l3_share_for_l3": 0.20}
VARIANT = re.compile(r"\s*[\(\[][^\)\]]*(?:learning|synchronous|asynchronous|classroom|online|blended|e-?learn|run|batch|intake)[^\)\]]*[\)\]]", re.I)
COMPILED = {k: re.compile(v["pattern"], re.I) for k, v in TASKS.items()}


def quote(text: str, match: re.Match, radius: int = 90) -> str:
    start, end = max(0, match.start()-radius), min(len(text), match.end()+radius)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def match_tasks(outcomes: str, description: str) -> list[dict]:
    """Tasks a course teaches. Outcome text is stronger evidence than a general description."""
    found = []
    for task, rx in COMPILED.items():
        for field, text in (("outcomes", outcomes), ("description", description)):
            m = rx.search(text) if text else None
            if m:
                found.append({"task": task, "evidence_field": field, "quote": quote(text, m)})
                break
    return found


def sector_of(title: str) -> str:
    for sector, (title_rx, _) in SECTORS.items():
        if re.search(title_rx, title or "", re.I):
            return sector
    return CROSS


def course_frame(cet: pd.DataFrame, uni: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    cet = cet.assign(title=cet.coursetitle, provider_name=cet.trainingprovideralias, hours=cet.number_of_hours,
                     attendance=cet.attendancecount, source="SkillsFuture", outcome_text=cet.what_you_learn,
                     description=cet.about_this_course)
    uni = uni.assign(title=uni.course_name, provider_name=uni.provider, hours=np.nan, attendance=np.nan,
                     source="University", outcome_text=uni.learning_outcomes, description=uni.synopsis)
    cols = ["record_id", "title", "provider_name", "hours", "attendance", "source", "outcome_text", "description", "AI Categories"]
    c = pd.concat([cet[cols], uni[cols]], ignore_index=True).merge(labels[["record_id", "strict_ai", "expanded_ai"]], on="record_id", how="left")
    # "No AI yet" is conservative: no AI under the expanded screen and no dictionary AI category.
    c["has_ai"] = c.expanded_ai.fillna(False).astype(bool) | c.strict_ai.fillna(False).astype(bool) | c["AI Categories"].notna()
    c["sector"] = c.title.map(sector_of)
    c["outcomes"] = c.outcome_text.map(usable_outcomes)
    c["description"] = c.description.map(clean)
    return c


def merge_variants(courses: pd.DataFrame) -> pd.DataFrame:
    """One row per course offered in several formats; attendance is summed across them."""
    c = courses.assign(base_title=courses.title.fillna("").map(lambda t: re.sub(r"\s+", " ", VARIANT.sub("", t)).strip()))
    c["_key"] = c.base_title.str.casefold() + "|" + c.provider_name.fillna("").str.casefold()
    c["_order"] = c.outcomes.str.len().fillna(0)
    first = c.sort_values("_order", ascending=False).drop_duplicates("_key").set_index("_key")
    first["attendance"] = c.groupby("_key").attendance.sum(min_count=1)
    first["variants"] = c.groupby("_key").size()
    first["variant_ids"] = c.groupby("_key").record_id.agg(lambda s: "; ".join(sorted(s)))
    first["has_ai"] = c.groupby("_key").has_ai.any()
    return first.reset_index(drop=True).drop(columns=["_order"])


def task_demand(jobs: pd.DataFrame, job_labels: pd.DataFrame) -> pd.DataFrame:
    """Per sector and task: ads describing the task, how many ask for AI, which AI skills
    those AI-asking ads mention and at what level. The cross-industry row uses all ads."""
    j = jobs[["record_id", "Category", "review_text"]].merge(job_labels[["record_id", "strict_ai", "maturity"]], on="record_id", how="left")
    j["strict_ai"] = j.strict_ai.fillna(False).astype(bool)
    hits = pd.DataFrame({t: j.review_text.str.contains(rx, regex=True) for t, rx in COMPILED.items()})
    groups = {s: j.Category.fillna("").str.contains(cat_rx, case=False, regex=True) for s, (_, cat_rx) in SECTORS.items()}
    groups[CROSS] = pd.Series(True, index=j.index)
    rows = []
    for sector, in_sector in groups.items():
        n_sector = int(in_sector.sum())
        for task in TASKS:
            hit = in_sector & hits[task]
            asking = j[hit & j.strict_ai]
            skills = {k: int(asking.review_text.str.contains(v, case=False, regex=True).sum()) for k, v in SKILLS.items()}
            top = [f"{k} ({n})" for k, n in sorted(skills.items(), key=lambda x: -x[1]) if n >= 3][:3]
            n, a = int(hit.sum()), len(asking)
            rows.append({"sector": sector, "task": task, "sector_jobs": n_sector, "task_jobs": n,
                         "task_share": n/n_sector if n_sector else np.nan, "task_jobs_asking_ai": a,
                         "ai_share_of_task_jobs": a/n if n else np.nan,
                         "l3_plus_share_of_ai_asking": float(asking.maturity.ge(3).mean()) if a else np.nan,
                         "top_ai_skills_in_ai_asking_ads": "; ".join(top)})
    return pd.DataFrame(rows)


def upgrade_tier(task: str, dm: pd.Series) -> str:
    """Declared rule for the next step beyond the L1 baseline."""
    if not dm.task_share >= RULES["min_task_share_for_l2"]:
        return ""
    if TASKS[task]["level"] == "L3":
        backed = dm.task_jobs_asking_ai >= RULES["min_ai_task_ads_for_l3"] and dm.l3_plus_share_of_ai_asking >= RULES["min_l3_share_for_l3"]
        return "L3 hiring-backed" if backed else "L3 build ahead"
    return "L2 task upgrade"


def baseline_text(sector: str, task: str | None) -> str:
    if task:
        return f"AI awareness for {TASKS[task]['label'].lower()}: where AI helps, how to check it, keeping data safe"
    return BASELINE[sector]


def suggestions(courses: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """One row per non-AI course: an L1 baseline always, plus the best-supported upgrade.
    Upgrade choice (declared): outcome evidence before description, then task share of ads."""
    d = demand.set_index(["sector", "task"])
    rows = []
    for _, c in courses[~courses.has_ai].iterrows():
        best = None
        for m in match_tasks(c.outcomes or "", c.description or ""):
            dm = d.loc[(c.sector, m["task"])]
            key = (m["evidence_field"] == "outcomes", dm.task_share)
            if best is None or key > best[0]:
                best = (key, m, dm)
        row = {"record_id": c.record_id, "variant_ids": c.variant_ids, "variants": int(c.variants),
               "title": c.base_title, "provider": c.provider_name, "source": c.source, "sector": c.sector,
               "hours": c.hours, "attendance": c.attendance, "has_usable_outcomes": bool(c.outcomes),
               "baseline_level": "L1", "baseline_hours": BASELINE_HOURS,
               "baseline_basis": "Policy: every course gets AI literacy for its field; not a demand claim",
               "task": "", "task_label": "", "evidence_field": "", "course_quote": "", "task_share_of_sector_jobs": np.nan,
               "ai_share_of_task_jobs": np.nan, "top_ai_skills_in_ai_asking_ads": "", "upgrade_tier": "",
               "upgrade": "", "upgrade_hours": np.nan}
        if best:
            _, m, dm = best
            t = TASKS[m["task"]]
            tier = upgrade_tier(m["task"], dm)
            row.update(task=m["task"], task_label=t["label"], evidence_field=m["evidence_field"], course_quote=m["quote"],
                       task_share_of_sector_jobs=dm.task_share, ai_share_of_task_jobs=dm.ai_share_of_task_jobs,
                       top_ai_skills_in_ai_asking_ads=dm.top_ai_skills_in_ai_asking_ads, upgrade_tier=tier,
                       upgrade=t["addon"] if tier else "", upgrade_hours=t["hours"] if tier else np.nan)
        row["baseline"] = baseline_text(c.sector, row["task"] or None)
        if HANDS_ON.search(c.base_title or ""):
            row["baseline_basis"] = "Optional: hands-on or regulated course; review fit with the provider and regulator"
        rows.append(row)
    return pd.DataFrame(rows)


def summary(courses: pd.DataFrame, sugg: pd.DataFrame) -> dict:
    res = {}
    for sector in [*SECTORS, CROSS]:
        s = courses[courses.sector.eq(sector)]
        g = sugg[sugg.sector.eq(sector)]
        res[sector] = {"courses_after_merging_variants": len(s), "with_ai": int(s.has_ai.sum()), "no_ai": len(g),
                       "no_ai_attendance": int(g.attendance.fillna(0).sum()),
                       "l1_baseline": len(g),
                       "l1_optional_hands_on": int(g.baseline_basis.str.startswith("Optional").sum()),
                       **{tier: int(g.upgrade_tier.eq(tier).sum()) for tier in ("L2 task upgrade", "L3 hiring-backed", "L3 build ahead")},
                       "upgrade_from_outcomes": int((g.upgrade_tier.ne("") & g.evidence_field.eq("outcomes")).sum()),
                       "upgrade_from_description_only": int((g.upgrade_tier.ne("") & g.evidence_field.eq("description")).sum()),
                       "without_usable_outcomes": int((~g.has_usable_outcomes).sum())}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUTPUT / "infusion"))
    args = ap.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    cet, uni = load_courses()
    courses = merge_variants(course_frame(cet, uni, pd.read_csv(OUTPUT / "course_labels.csv")))
    demand = task_demand(load_jobs(), pd.read_csv(OUTPUT / "job_labels.csv", usecols=["record_id", "strict_ai", "maturity"]))
    sugg = suggestions(courses, demand)
    demand.to_csv(dest / "task_demand.csv", index=False)
    sugg.to_csv(dest / "integration_suggestions.csv", index=False)
    meta = {"sectors": summary(courses, sugg), "rules": RULES, "baseline_hours": BASELINE_HOURS,
            "tasks": {k: {x: v[x] for x in ("label", "addon", "level", "hours")} for k, v in TASKS.items()},
            "limitations": ["L1 baseline is a policy choice for every course, not evidence of demand",
                            "Lexical task matches on published text, not audited syllabi",
                            "Course industry from title keywords; untagged courses are compared with all job ads",
                            "Attendance is the MySkillsFuture attendancecount summed across format variants; universities have none",
                            "Add-on wording is a declared template; skills and level evidence come from AI-asking job ads"]}
    (dest / "summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["sectors"], indent=2))


if __name__ == "__main__":
    main()
