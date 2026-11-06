"""Plain-English learning objectives for Gapwise, each tied to the evidence behind it.

Objective wording comes from declared templates (one per skill or workplace task, with a
"use AI" and a "build with AI" form); which objectives appear, and the evidence shown with
them, come from the verdict engine, course_infusion matches and job ads. Nothing here calls
a model. Plain ratings (High / Medium / Low) use declared cut-offs listed in RATINGS.
"""
from __future__ import annotations
import re

DEPTHS = {"understand": "Understand what AI can do", "use": "Use AI tools in their work", "build": "Build things with AI"}
# Engine depth interval for each plain choice (AI Depth Ladder levels).
DEPTH_LEVELS = {"understand": (1, 2), "use": (2, 3), "build": (3, 4)}
WORK = {"Banking and Finance": "finance", "Accounting / Auditing / Taxation": "accounting", "Insurance": "insurance",
        "Human Resources": "HR", "Logistics / Supply Chain": "logistics", "Building and Construction": "construction",
        "Customer Service": "customer service", "Marketing / Public Relations": "marketing", "Sales / Retail": "sales",
        "Healthcare / Pharmaceutical": "healthcare", "Admin / Secretarial": "office", "Information Technology": "software",
        "Engineering": "engineering", "Consulting": "consulting", "Manufacturing": "manufacturing",
        "Education and Training": "education", "Advertising / Media": "media", "Design": "design"}
# Plain names for the engine's course concepts.
CONCEPTS = {"assurance": "Using AI responsibly at work", "productivity": "AI for everyday work", "documents": "AI that answers from your documents",
            "language": "AI for customer messages and text", "analytics": "AI for data analysis and forecasting",
            "workflow": "Automating work with AI assistants", "prediction": "Predicting outcomes with AI",
            "vision": "AI for images and inspection", "production": "Running AI systems reliably"}
PLAIN_SKILL = {"LLM": "AI chat tools", "RAG": "AI that answers from your documents", "AI agents": "AI assistants that carry out tasks",
               "Responsible AI": "safe and fair AI use", "Generative AI": "generative AI", "Machine learning": "machine learning",
               "Python": "Python", "SQL": "SQL", "MLOps": "running AI models in production", "Cloud": "cloud tools",
               "NLP": "AI for text", "Computer vision": "AI for images", "Deep learning": "deep learning",
               "Data engineering": "data pipelines"}
SKILL_LO = {
    "Generative AI": ("Use generative AI to draft and improve {work} documents, and check every output for accuracy before sharing it",
                      "Design a generative AI workflow for a {work} task and test it against agreed quality criteria"),
    "LLM": ("Write clear instructions for an AI chat tool to complete a {work} task, and spot answers that are wrong or made up",
            "Connect a language model to a {work} task and measure how often its answers are correct"),
    "RAG": ("Use an AI assistant that answers from approved {work} documents, and check each answer against its source",
            "Build an AI assistant that answers questions from approved {work} documents and shows where each answer came from"),
    "AI agents": ("Set up an AI assistant to carry out a multi-step {work} task, with clear points where a person must approve",
                  "Design an AI agent for a {work} workflow with defined tools, stop rules and human approval"),
    "Responsible AI": ("Assess an AI use in {work} for privacy, bias and accountability risks, and decide what checks it needs",) * 2,
    "Machine learning": ("Interpret the results of a machine-learning model for a {work} decision, including its limits",
                         "Train and evaluate a machine-learning model for a {work} problem against a simple baseline"),
    "Python": ("Clean and analyse {work} data in Python with help from an AI coding assistant",
               "Write and test Python code that automates a {work} data task"),
    "SQL": ("Query and check {work} data with SQL so it is ready for AI analysis",) * 2,
    "MLOps": ("Explain how AI models are monitored once in use, and when to raise a problem",
              "Deploy and monitor an AI model for {work}, with a plan for when it fails"),
    "Cloud": ("Use cloud AI tools securely, with the right access and data controls",
              "Set up a secure cloud environment for an AI workload, with access and cost controls"),
    "NLP": ("Use AI to sort and summarise {work} text such as emails, feedback or reports, and review its mistakes",
            "Build and evaluate an AI text classifier for {work} messages"),
    "Computer vision": ("Evaluate an AI image tool for a {work} inspection task and decide when a person must check",
                        "Train and test an image model for a {work} inspection task"),
    "Deep learning": ("Explain how deep-learning models work and when they suit a {work} problem",
                      "Fine-tune a deep-learning model for a {work} problem and report its errors"),
    "Data engineering": ("Describe how data flows into AI tools in {work}, and where quality problems arise",
                         "Build a reliable data pipeline that feeds an AI tool in {work}"),
}
TASK_LO = {
    "report_writing": "Draft {work} reports with AI help, then fact-check and edit them before use",
    "budgeting_forecasting": "Prepare budgets and forecasts with AI help, and explain the variances in plain terms",
    "financial_statements": "Analyse financial statements with AI help and check its answers against the source figures",
    "document_review": "Use AI to find answers in contracts, policies and regulations, and confirm each answer in the source",
    "audit_reconciliation": "Use AI to flag unusual entries and reconciliation gaps, and decide which need a person's review",
    "data_analysis": "Analyse spreadsheet and dashboard data with an AI assistant, and check its calculations",
    "customer_communication": "Draft replies to customer enquiries with AI, and decide when a case must go to a person",
    "research_summarising": "Research and summarise information with AI, citing the sources it used",
    "content_marketing": "Create marketing content with generative AI that meets brand, accuracy and copyright rules",
    "recruitment": "Use AI to help screen candidates, and check its suggestions for fairness and bias",
    "tender_procurement": "Review tender documents and specifications with AI, and verify key terms by hand",
    "planning_scheduling": "Plan schedules and inventory with AI suggestions, and test them against real constraints",
    "inspection_quality": "Use AI image tools to support quality checks, and know when a person must inspect",
}
BASELINE_LO = "Explain what AI can and cannot do in {work}, and follow safe practices for confidential data"
# Key ideas an AI rewrite must keep, one regex per idea (alternatives with |). They tie the
# objective to the evidence shown beside it, so a rewrite cannot drift to another skill.
SKILL_ANCHORS = {"Generative AI": ["generative|genai"], "LLM": ["chat|language model|llm|prompt|instruct", "wrong|made up|incorrect|error|inaccura|correct"],
                 "RAG": ["document|polic|manual|knowledge", "source|came from|cite|citation"], "AI agents": ["agent|assistant", "approv|sign-off|human"],
                 "Responsible AI": ["privacy|bias|accountab|risk"], "Machine learning": ["model"], "Python": ["python"], "SQL": ["sql"],
                 "MLOps": ["monitor"], "Cloud": ["cloud"], "NLP": ["text|email|feedback|message"], "Computer vision": ["image|inspect"],
                 "Deep learning": ["deep"], "Data engineering": ["data"]}
TASK_ANCHORS = {"report_writing": ["report", "check|edit|fact|review"], "budgeting_forecasting": ["budget|forecast"],
                "financial_statements": ["statement"], "document_review": ["contract|polic|regulat|document"],
                "audit_reconciliation": ["reconcil|unusual|anomal|audit"], "data_analysis": ["data|spreadsheet|dashboard|excel"],
                "customer_communication": ["customer|enquir|client"], "research_summarising": ["research|summar"],
                "content_marketing": ["content|marketing|campaign|post"], "recruitment": ["candidate|recruit|screen"],
                "tender_procurement": ["tender|specification|purchase|procure|vendor"],
                "planning_scheduling": ["schedul|roster|inventory|plan"], "inspection_quality": ["inspect|quality"]}
BASELINE_ANCHORS = ["confidential|privacy|sensitive|data", "cannot|can't|limit|falls short|short"]
RATINGS = {"demand_top_share": 0.25, "demand_mid_share": 0.75,          # sector rank by AI job ads
           "competition_low": 30, "competition_high": 90,              # AI courses per 100 AI job ads
           "future_high": 0.35, "future_mid": 0.25}                    # share of jobs with AI-ready tasks
NOW, GAP, NEXT, ALL = "Employers ask for this now", "Few courses teach this", "Where the work is heading", "Everyone needs this"


def work_word(sector: str) -> str:
    return WORK.get(sector, re.sub(r"\s*/.*", "", sector).lower())


def guess_depth(text: str, default: str = "use") -> str:
    """Depth the user's own words ask for; `default` (the industry evidence) when they name none."""
    t = text.lower()
    if re.search(r"\b(build|develop|code|coding|python|program|engineer|deploy|train (?:a|the) model)", t):
        return "build"
    if re.search(r"\b(awareness|introduc|basics|literacy|understand|overview|non-technical)", t):
        return "understand"
    return default


LEARNER_PATTERNS = [r"\bfor ([a-z][a-z &/-]{2,40}?)(?:\s+(?:to|who|on|in|about|with|so)\b|[,.]|$)",
                    r"\b(?:teach(?:es|ing)?|train(?:s|ing)?|help(?:s|ing)?|equip(?:s|ping)?) ([a-z][a-z &/-]{2,40}?)\s+(?:to|how|on|in|with)\b",
                    r"\bwhere ([a-z][a-z &/-]{2,40}?)\s+(?:build|learn|use|create|develop)\b"]


def guess_learners(text: str) -> str:
    for pattern in LEARNER_PATTERNS:
        m = re.search(pattern, text.lower())
        if m and m.group(1).strip() not in ("you", "them", "people", "my staff"):
            return m.group(1).strip()
    return "working professionals"


def lo(template: str, sector: str) -> str:
    return template.format(work=work_word(sector))


def for_new_course(sector: str, depth: str, verdict: dict, open_skills: list[str], course_counts: dict, ai_courses: int,
                   ahead_tasks: list[dict]) -> list[dict]:
    """Objectives for a new idea: baseline, the engine's best-supported skills, one open-space
    skill and one forward-looking task. Each carries its evidence tag and sentence."""
    i = 0 if depth != "build" else 1
    out = [{"text": lo(BASELINE_LO, sector), "tag": ALL, "why": "AI literacy for the field, whatever the course level", "anchors": BASELINE_ANCHORS}]
    mentions = {r["skill"]: r["sector_job_mentions"] for r in verdict.get("skill_evidence", [])}
    skills = [s for s in verdict.get("skills", []) if s in SKILL_LO]
    if depth == "understand":
        skills = [s for s in ["Responsible AI", *skills] if s in SKILL_LO][:2]
    for s in dict.fromkeys(skills):
        n = mentions.get(s)
        asked = f"{n:,} {work_word(sector)} job ads that ask for AI mention {PLAIN_SKILL[s]}" if n else f"Part of the best-supported idea for {work_word(sector)}"
        if s in open_skills:
            # Demanded and rarely taught: the strongest case, so say both.
            out.append({"text": lo(SKILL_LO[s][i], sector), "tag": GAP, "anchors": SKILL_ANCHORS[s],
                        "why": f"{asked}, but only {course_counts.get(s, 0)} of {ai_courses} {work_word(sector)} AI courses do"})
        else:
            out.append({"text": lo(SKILL_LO[s][i], sector), "tag": NOW, "why": asked, "anchors": SKILL_ANCHORS[s]})
    for s in [s for s in open_skills if s in SKILL_LO and s not in skills][:1]:
        out.append({"text": lo(SKILL_LO[s][i], sector), "tag": GAP, "anchors": SKILL_ANCHORS[s],
                    "why": f"Only {course_counts.get(s, 0)} of {ai_courses} {work_word(sector)} AI courses mention {PLAIN_SKILL[s]}"})
    for t in ahead_tasks[:1]:
        if t["task"] in TASK_LO:
            out.append({"text": lo(TASK_LO[t["task"]], sector), "tag": NEXT, "anchors": TASK_ANCHORS[t["task"]],
                        "why": f"{t['task_share']:.0%} of {work_word(sector)} jobs involve this, but only {t['ai_share']:.0%} of those ads ask for AI yet"})
    return out


def for_existing_course(sector: str, depth: str, matches: list[dict], demand: dict) -> list[dict]:
    """Objectives to add to a course someone already runs: a baseline, then the workplace tasks
    their own description teaches, most common in job ads first. demand: task -> (share, ai_share)."""
    out = [{"text": lo(BASELINE_LO, sector), "tag": ALL, "why": "AI literacy for the field, whatever the course covers", "anchors": BASELINE_ANCHORS}]
    ranked = sorted(matches, key=lambda m: -demand.get(m["task"], (0, 0))[0])
    for m in ranked[:3]:
        share, ai = demand.get(m["task"], (0, 0))
        tag = NEXT if ai < 0.05 else NOW
        out.append({"text": lo(TASK_LO[m["task"]], sector), "tag": tag, "anchors": TASK_ANCHORS[m["task"]],
                    "why": f"Your course already covers this. {share:.0%} of {work_word(sector)} job ads describe it; {ai:.0%} of those ask for AI",
                    "quote": m["quote"]})
    if depth == "build" and len(out) < 5:
        out.append({"text": lo(SKILL_LO["AI agents"][1], sector), "tag": NEXT, "why": "Chosen because learners will build with AI",
                    "anchors": SKILL_ANCHORS["AI agents"]})
    return out


def rate(value, low, high, reverse=False):
    if value is None:
        return "Unknown"
    level = "High" if value >= high else ("Medium" if value >= low else "Low")
    return {"High": "Low", "Low": "High"}.get(level, level) if reverse else level


def demand_rating(rank: int, total: int) -> str:
    share = rank / max(total, 1)
    return "High" if share <= RATINGS["demand_top_share"] else ("Medium" if share <= RATINGS["demand_mid_share"] else "Low")


def plain_text(objectives: list[dict], title: str) -> str:
    lines = [title, "", "By the end of this course, learners will be able to:"]
    lines += [f"{n}. {o['text']}." for n, o in enumerate(objectives, 1)]
    lines += ["", "Evidence behind each objective:"] + [f"{n}. {o['tag']}: {o['why']}." for n, o in enumerate(objectives, 1)]
    lines += ["", "Drafted by Gapwise from Singapore job ads and course catalogues. Review before use."]
    return "\n".join(lines)
