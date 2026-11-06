"""Gapwise prototype: plain-English learning objectives for an AI course, with evidence.

Screens: Home (describe it) → path (new course or add AI to one you run) → who → depth →
result, plus Radar (industries to build ahead in) and AI key settings.
Run: python -m streamlit run gapwise_app.py --server.port 8504
Verdicts come from the six-rule engine; objective wording from gapwise_objectives.
"""
import html
import os
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gapwise", page_icon="↗", layout="wide", initial_sidebar_state="collapsed")

# Hosted copy: the DeepSeek key comes only from the owner's Streamlit Secrets, never from visitors.
try:
    if "DEEPSEEK_API_KEY" in st.secrets:
        os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
except Exception:  # no secrets file: run in local mode
    pass

import gapwise_data as g
import gapwise_objectives as o
from course_infusion import TASKS, match_tasks
from provider_decisions import DESIGNS
from deepseek_client import KEY_PATH, credential
from agentic.transport import AgentBudget


def agent_available():
    try:
        return bool(credential()) and AgentBudget(2).spent() < 2
    except RuntimeError:
        return False

INK, VIOLET, CORAL, TEAL, AMBER, MUTED = "#0B0D17", "#5B3FE0", "#FF6B57", "#14B8A6", "#D97706", "#4A4E63"
TAG_STYLE = {o.NOW: ("#F1EEFF", "#3A2A9E"), o.GAP: ("#FFE9E5", "#8A2A1B"), o.NEXT: ("#E2F6F3", "#0A5F55"), o.ALL: ("#F1F2F7", "#3B3F52")}
KEYWORDS = {"financ": "Banking and Finance", "bank": "Banking and Finance", "account": "Accounting / Auditing / Taxation",
            "audit": "Accounting / Auditing / Taxation", " hr ": "Human Resources", "human resource": "Human Resources", "recruit": "Human Resources",
            "logistic": "Logistics / Supply Chain", "supply chain": "Logistics / Supply Chain", "market": "Marketing / Public Relations",
            "retail": "Sales / Retail", "sales": "Sales / Retail", "customer": "Customer Service", "health": "Healthcare / Pharmaceutical",
            "nurs": "Healthcare / Pharmaceutical", "construct": "Building and Construction", "engineer": "Engineering",
            "manufactur": "Manufacturing", "software": "Information Technology", " it ": "Information Technology",
            "consult": "Consulting", "admin": "Admin / Secretarial", "teach": "Education and Training", "educat": "Education and Training"}
# Short names for the six engine checks, as on the result card.
CHECK_NAMES = {"Observed hiring signal": "Real demand", "Sector course / job ratio": "Low supply",
               "Audience connects to real job titles": "Clear learner", "Depth matches the brief and sector": "Right depth",
               "Measured skill-mention gap": "Skill gap", "Brief fit and actual outcome evidence": "Stands out"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stApp p, .stApp div, .stApp span, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp li, .stApp h1, .stApp h2, .stApp h3
  {{font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;}}
.stApp {{background: #F5F6FA; color: {INK};}}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="InputInstructions"] {{display: none !important;}}
.block-container {{max-width: 1160px; padding: 1.4rem 1.5rem 5rem;}}
h1 {{letter-spacing: -0.055em; color: {INK}; font-weight: 800 !important; line-height: 1.02 !important;}}
p {{color: {MUTED};}}

/* Buttons: secondary = white pill, primary = ink. */
.stButton button, .stDownloadButton > button {{border-radius: 999px; min-height: 44px; font-weight: 600; border: 0 !important; background: #fff; color: {INK};
  box-shadow: 0 1px 2px rgba(11,13,23,.06), 0 0 0 1px rgba(11,13,23,.05); padding: 0 18px;}}
.stButton > button:hover, .stDownloadButton > button:hover {{color: {VIOLET}; box-shadow: 0 2px 8px rgba(91,63,224,.15), 0 0 0 1px rgba(91,63,224,.3);}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{background: {INK}; color: #fff; border-radius: 14px; min-height: 52px; font-weight: 700; padding: 0 26px;}}
.stButton > button[kind="primary"]:hover {{background: #24283C; color: #fff;}}
.stButton > button:disabled {{opacity: .35;}}
.stButton > button p, .stDownloadButton > button p {{color: inherit; font-weight: inherit;}}

/* Inputs: white, rounded, no grey fill. */
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {{background: #fff !important; border: 0 !important; border-radius: 16px !important;}}
.stTextArea textarea, .stTextInput input {{background: #fff !important; font-size: 1rem; color: {INK};}}
.stTextArea > div, .stTextInput > div > div, [data-baseweb="select"] > div {{box-shadow: 0 1px 2px rgba(11,13,23,.06), 0 0 0 1px rgba(11,13,23,.06); border-radius: 16px;}}
.stTextArea label p, .stTextInput label p, .stSelectbox label p {{font-weight: 600; color: {MUTED}; font-size: .85rem;}}
[data-testid="stExpander"] details {{background: #fff; border: 0; border-radius: 16px; box-shadow: 0 0 0 1px rgba(11,13,23,.05);}}
[data-testid="stExpander"] summary p {{font-weight: 600; color: {INK};}}

/* Header: logo, nav pill, key pill. */
[data-testid="column"]:has(.gw-logo) button, [data-testid="column"]:has(.gw-logo) button:focus {{background: transparent; box-shadow: none !important; font-size: 1.25rem; font-weight: 800; letter-spacing: -.04em; padding: 0; min-height: 40px;}}
[data-testid="column"]:has(.gw-logo) button p {{font-size: 1.25rem; font-weight: 800 !important; color: #0B0D17;}}
[data-testid="column"]:has(.gw-logo) button::before {{content: "↗"; display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px;
  border-radius: 10px; background: {INK}; color: #fff; font-size: 1rem; margin-right: 10px;}}
[data-testid="column"]:has(.gw-logo) button:hover {{box-shadow: none;}}
[data-testid="column"]:has(.gw-nav) [data-testid="stHorizontalBlock"] {{background: #fff; border-radius: 999px; padding: 4px; gap: 2px; width: fit-content; margin: 0 auto;
  box-shadow: 0 1px 2px rgba(11,13,23,.06), 0 0 0 1px rgba(11,13,23,.05);}}
[data-testid="column"]:has(.gw-nav) [data-testid="stHorizontalBlock"] > div {{width: auto !important; flex: none !important; min-width: 0;}}
[data-testid="column"]:has(.gw-nav) button {{box-shadow: none; min-height: 38px; border-radius: 999px !important; padding: 0 18px; font-weight: 600;}}
[data-testid="column"]:has(.gw-nav) button[kind="primary"] {{min-height: 38px; background: {INK};}}
[data-testid="column"]:has(.gw-key) .stButton {{display: flex; justify-content: flex-end;}}
[data-testid="column"]:has(.gw-key) button {{font-size: .82rem; min-height: 40px;}}

/* Search bar: text area with a violet arrow button. */
[data-testid="stHorizontalBlock"]:has(.gw-arrow):not(:has([data-testid="stHorizontalBlock"] .gw-arrow)) {{background: #fff; border-radius: 22px; padding: 8px 8px 8px 6px; align-items: center;
  box-shadow: 0 10px 30px rgba(11,13,23,.07), 0 0 0 1px rgba(11,13,23,.05);}}
[data-testid="stHorizontalBlock"]:has(.gw-arrow):not(:has([data-testid="stHorizontalBlock"] .gw-arrow)) .stTextArea > div {{box-shadow: none;}}
[data-testid="stHorizontalBlock"]:has(.gw-arrow) textarea {{resize: none; font-size: 1.1rem; padding-top: 14px;}}
[data-testid="column"]:has(.gw-arrow):not(:has([data-testid="column"] .gw-arrow)) button {{width: 56px; height: 56px; min-height: 56px; border-radius: 16px !important; background: {VIOLET} !important; color: #fff;
  font-size: 1.3rem; padding: 0; float: right;}}
[data-testid="column"]:has(.gw-arrow):not(:has([data-testid="column"] .gw-arrow)) button:disabled {{background: #C9C2F5 !important; opacity: 1;}}

/* Whole-card click: an invisible button laid over the card drawn just before it. */
div.element-container:has(.gw-hit) + div.element-container {{position: relative; z-index: 3;}}
div.element-container:has(.gw-hit) + div.element-container .stButton {{width: 100%;}}
div.element-container:has(.gw-hit) + div.element-container button {{opacity: 0; width: 100%; border-radius: 22px !important;}}
div.element-container:has(.gw-hit-row) + div.element-container {{margin-top: -96px; height: 96px;}}
div.element-container:has(.gw-hit-row) + div.element-container button {{height: 96px;}}
div.element-container:has(.gw-hit-tile) + div.element-container {{margin-top: -160px; height: 160px;}}
div.element-container:has(.gw-hit-tile) + div.element-container button {{height: 160px;}}
div.element-container:has(.gw-hit-depth) + div.element-container {{margin-top: -240px; height: 240px;}}
div.element-container:has(.gw-hit-depth) + div.element-container button {{height: 240px;}}
div.element-container:has(.gw-hit):has(+ div.element-container button:hover) .gw-card {{box-shadow: 0 12px 30px rgba(91,63,224,.16), 0 0 0 2px {VIOLET}; transform: translateY(-1px);}}

.gw-card {{background: #fff; border-radius: 24px; box-shadow: 0 1px 2px rgba(11,13,23,.05), 0 0 0 1px rgba(11,13,23,.04); transition: all .15s;}}
.gw-eyebrow {{font-size: .82rem; font-weight: 700; color: {VIOLET}; margin-bottom: 4px;}}
.gw-pill {{display: inline-block; border-radius: 999px; padding: 4px 10px; font-size: .72rem; font-weight: 800; letter-spacing: .03em;}}
.gw-help {{color: {MUTED}; font-size: 1.05rem;}}
.gw-lo {{display: flex; gap: 16px; padding: 18px 0; border-top: 1px solid #EEF0F5;}}
.gw-lo:first-of-type {{border-top: 0;}}
.gw-num {{flex: none; width: 34px; height: 34px; border-radius: 50%; background: #F1EEFF; color: {VIOLET}; display: flex; align-items: center; justify-content: center; font-weight: 800;}}
.gw-lo-text {{font-size: 1.06rem; font-weight: 700; color: {INK}; line-height: 1.45; letter-spacing: -.01em;}}
.gw-why {{color: {MUTED}; font-size: .88rem; margin-top: 6px;}}
.gw-quote {{background: #F5F6FA; border-radius: 12px; padding: 8px 12px; font-size: .86rem; color: #2A2E40; margin-top: 8px;}}
.gw-chip {{display: inline-flex; gap: 6px; align-items: center; background: #1A1D2E; color: #E6E7EE; border-radius: 999px; padding: 7px 13px; margin: 4px 6px 0 0; font-size: .82rem; font-weight: 600;}}
.gw-step-chip {{display: inline-block; background: #F1F2F7; color: #3B3F52; border-radius: 999px; padding: 6px 12px; margin: 3px 4px; font-size: .82rem; font-weight: 500;}}
[data-testid="column"]:has(.gw-dark) {{background: {INK}; border-radius: 28px; padding: 26px 26px 22px;}}
[data-testid="column"]:has(.gw-dark) button {{background: #fff; color: {INK}; border-radius: 14px !important; min-height: 52px; font-weight: 700; width: 100%;}}
[data-testid="column"]:has(.gw-dark) label p {{color: #B7BACB;}}
@media (max-width: 640px) {{ .gw-h1 {{font-size: 2.4rem !important;}} }}
</style>
"""


def esc(x):
    return html.escape(str(x))


def md(s):
    st.markdown(s, unsafe_allow_html=True)


def pill(text, bg, fg):
    return f'<span class="gw-pill" style="background:{bg};color:{fg}">{esc(text)}</span>'


def go(screen, **kw):
    st.session_state.screen = screen
    st.session_state.update(kw)


def restart():
    for k in [k for k in st.session_state if k not in ("screen",)]:
        del st.session_state[k]
    st.session_state.screen = "start"


def guess_sector(text):
    t = f" {text.lower()} "
    return next((s for k, s in KEYWORDS.items() if k in t), None)


def sector_names():
    return g.sectors().sector.tolist()


def label(sector):
    return sector.replace(" / ", ", ")


def short(sector):
    return sector.split(" / ")[0].replace("Information Technology", "IT").replace("Banking and Finance", "Finance")


@st.cache_data(show_spinner=False)
def cached_verdict(sector, audience, need, lmin, lmax):
    return g.verdict(sector, audience, need, lmin, lmax, 24)


@st.cache_data(show_spinner=False)
def cached_ideas(sector):
    return g.ideas(sector)


def header():
    screen = st.session_state.screen
    a, b, c = st.columns([1.3, 2, 1.3])
    with a:
        md('<span class="gw-logo"></span>')
        st.button("Gapwise", key="logo", on_click=restart, help="Start again")
    with b:
        md('<span class="gw-nav"></span>')
        n1, n2 = st.columns(2)
        on_radar = screen == "radar"
        n1.button("Home", key="nav_home", type="secondary" if on_radar else "primary", on_click=restart)
        n2.button("Radar", key="nav_radar", type="primary" if on_radar else "secondary", on_click=go, args=("radar",))
    with c:
        md('<span class="gw-key"></span>')
        if screen == "settings":
            st.button("← Back", key="from_settings", on_click=go, args=(st.session_state.get("back_to", "start"),))
        else:
            live = agent_available()
            st.button("🟢 AI connected" if live else "⚪ Local mode · Add AI key", key="to_settings", on_click=go, args=("settings",),
                      kwargs={"back_to": screen})


# ---------------- 1. Home ----------------
EXAMPLE = ("A new course idea", "A course for finance analysts on using AI to answer questions from company policies")
UPGRADE_HINT = "Paste your course title and description, or its current learning outcomes"


def use_example(mode, text):
    st.session_state.mode = mode
    st.session_state.text = text


def upgrade_mode():
    st.session_state.mode = "A course I already run"
    st.session_state.text = ""


def ladder_card():
    lad = g.ladder()
    h = g.headline()
    top = max(lad.job_share.max(), lad.course_share.max())
    tiles = []
    for r in lad.itertuples():
        gap = r.job_share - r.course_share >= 0.05
        bar = lambda share, colour: (f'<div style="width:34px;height:{max(8, 180 * share / top):.0f}px;background:{colour};'
                                     f'border-radius:{10 if share > .04 else 6}px"></div>')
        tiles.append(f'<div style="flex:1;min-width:120px;background:{"#EEEBFF" if gap else "#F5F6FA"};border-radius:22px;padding:14px 10px 16px;'
                     f'display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:270px;box-sizing:border-box;position:relative">'
                     + (f'<div style="position:absolute;top:14px">{pill("GAP", INK, "#fff")}</div>' if gap else "")
                     + f'<div style="display:flex;gap:8px;align-items:flex-end">{bar(r.job_share, VIOLET)}{bar(r.course_share, CORAL)}</div>'
                     f'<div style="font-weight:700;margin-top:14px">{r.name}</div>'
                     f'<div style="font-size:.8rem;color:{MUTED}">L{r.maturity} · {r.job_share:.0%} vs {r.course_share:.0%}</div></div>')
    dot = lambda c, t: f'<span style="display:inline-flex;align-items:center;gap:6px;margin-left:18px"><span style="width:10px;height:10px;border-radius:50%;background:{c}"></span>{t}</span>'
    md(f'<div class="gw-card" style="padding:34px 36px;margin-top:34px">'
       f'<div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px">'
       f'<div><div class="gw-eyebrow">The big picture</div><div style="font-weight:800;font-size:2rem;letter-spacing:-.045em;line-height:1.1;max-width:560px">'
       f'Courses teach the basics. Jobs want people who can build with AI.</div></div>'
       f'<div style="font-size:.92rem;color:{INK}">{dot(VIOLET, "AI job ads")}{dot(CORAL, "AI courses")}</div></div>'
       f'<div style="display:flex;gap:14px;margin-top:26px;overflow-x:auto">{"".join(tiles)}</div>'
       f'<div style="font-size:.85rem;color:{MUTED};margin-top:16px">Share of {h["ai_jobs"]:,} AI job ads and {h["ai_courses"]:,} AI courses at each level of the '
       f'AI Depth Ladder. {h["courses_at_l1"]:.0%} of AI courses stop at awareness. Source: {h["jobs"]:,} MyCareersFuture job ads, SkillsFuture and university catalogues.</div></div>')


def start_screen():
    st.session_state.setdefault("mode", st.session_state.get("mode_val", "A new course idea"))
    keep("text")
    existing = st.session_state.mode == "A course I already run"
    md('<div style="text-align:center;margin-top:44px">'
       f'<span style="display:inline-flex;gap:8px;align-items:center;background:#fff;border-radius:999px;padding:5px 12px 5px 6px;font-size:.85rem;font-weight:500;'
       f'box-shadow:0 0 0 1px rgba(11,13,23,.05)">{pill("NEW", TEAL, "#fff")} Singapore AI skills scan, Sep 2026</span>'
       '<h1 class="gw-h1" style="font-size:4.2rem !important;margin:22px auto 14px;max-width:820px">What AI course should you build next?</h1>'
       '<p class="gw-help" style="max-width:600px;margin:0 auto 26px;font-size:1.15rem">Describe an idea, or paste a course you already run. Gapwise checks '
       'Singapore job ads and 38,630 existing courses, then writes learning objectives backed by the evidence.</p></div>')
    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        a, b = st.columns([12, 1])
        with a:
            text = st.text_area("Your course", key="text", height=68, label_visibility="collapsed",
                                placeholder=UPGRADE_HINT if existing else "A GenAI course for finance staff…")
        with b:
            md('<span class="gw-arrow"></span>')
            st.button("→", key="go_arrow", type="primary", disabled=len(text.strip()) < 15, on_click=understood, help="Continue")
        if existing:
            md(f'<div style="text-align:center;font-size:.9rem;color:{MUTED};margin-top:10px">Adding AI to a course you run. '
               'Include what learners do, e.g. “prepare budgets and monthly reports”.</div>')
        c = st.columns([1.15, 1.35, 1.15, 1.2])
        c[0].button("✦ Try an example", use_container_width=True, on_click=use_example, args=EXAMPLE)
        c[1].button("Show me the biggest gap", use_container_width=True, on_click=go, args=("ideas",))
        c[2].button("What's coming next?", use_container_width=True, on_click=go, args=("radar",))
        c[3].button("Upgrade my course", use_container_width=True, on_click=upgrade_mode)
    ladder_card()


def keep(*keys):
    """Streamlit drops a widget's value once it leaves the screen; restore it from the saved copy."""
    for k in keys:
        if k not in st.session_state and k + "_val" in st.session_state:
            st.session_state[k] = st.session_state[k + "_val"]


def save(*keys):
    for k in keys:
        if k in st.session_state:
            st.session_state[k + "_val"] = st.session_state[k]


def understood():
    save("text")
    st.session_state.mode_val = st.session_state.mode
    text = st.session_state.get("text", "")
    names = sector_names()
    guess = guess_sector(text)
    st.session_state.pop("sector", None)
    st.session_state.update(sector_val=guess if guess in names else names[0], sector_guessed=guess in names,
                            learners_val=o.guess_learners(text), depth_val=o.guess_depth(text), screen="path")


def confirmed(screen):
    save("sector")
    st.session_state.screen = screen


# ---------------- Ideas (optional) ----------------
def ideas_screen():
    md('<div style="margin-top:30px"><div class="gw-eyebrow">Course ideas</div>'
       '<h1 style="font-size:3rem !important">The biggest gaps, by industry</h1>'
       '<p class="gw-help">Ideas ranked by AI job demand, existing courses and skill gaps.</p></div>')
    names = sector_names()
    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        sector = st.selectbox("Industry", names, format_func=label, key="idea_sector")
        with st.spinner("Reading job ads and courses for this industry… (about 20 seconds the first time)"):
            ideas = cached_ideas(sector)
        for i, idea in enumerate(ideas):
            name = o.CONCEPTS.get(idea["key"], idea["concept"])
            what = DESIGNS[idea["key"]][1] if idea["key"] in DESIGNS else ""
            row_card(i + 1, name, what, f'{idea["jobs"]:,} AI JOB ADS', ("#F1EEFF", "#3A2A9E"), [VIOLET, CORAL, TEAL][i % 3])
            st.button("Use this idea", key=f"idea_{i}", on_click=use_idea, args=(sector, name, what))


def use_idea(sector, name, what):
    for k in ("mode", "text", "sector", "learners", "depth"):
        st.session_state.pop(k, None)
    st.session_state.update(mode_val="A new course idea", mode="A new course idea", text_val=f"{name}. {what}", sector_val=sector,
                            sector_guessed=True, learners_val="working professionals", depth_val="use", screen="who")


# ---------------- 2. Path, who and how deep ----------------
LEARNERS = ["Analysts", "Managers", "Operations staff", "Customer-facing staff", "Technical teams"]
DEPTH_TILES = {"understand": ("Understand", "Know what AI can and can't do", "L1–L2", 120),
               "use": ("Use", "Use AI tools well in their work", "L2–L3", 170), "build": ("Build", "Build AI into workflows and products", "L3–L4", 220)}
PERSON = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">'
          '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/></svg>')


def progress_bar(step):
    bars = "".join(f'<div style="flex:1;height:5px;border-radius:3px;background:{VIOLET if i <= step else "#E0E2EC"}"></div>' for i in range(3))
    text = st.session_state.text_val.strip()
    md(f'<div style="display:flex;gap:8px;max-width:320px;margin:26px auto 24px">{bars}</div>'
       f'<div style="text-align:center"><span style="display:inline-flex;gap:10px;align-items:center;background:#fff;border-radius:999px;padding:7px 16px 7px 8px;'
       f'font-size:.88rem;font-weight:600;box-shadow:0 0 0 1px rgba(11,13,23,.05);max-width:90%">'
       f'<span style="width:26px;height:26px;border-radius:50%;background:#F1EEFF;color:{VIOLET};display:inline-flex;align-items:center;justify-content:center">✦</span>'
       f'“{esc(text[:90] + ("…" if len(text) > 90 else ""))}”</span></div>')


def title(heading, sub):
    md(f'<h1 style="text-align:center;font-size:3rem !important;margin:24px 0 6px">{esc(heading)}</h1>'
       f'<p class="gw-help" style="text-align:center;margin-bottom:26px">{sub}</p>')


def pick(**kw):
    st.session_state.update({k + "_val": v for k, v in kw.items()})


def row_card(n, name, sub, badge, badge_style, colour, chosen=False):
    ring = f"box-shadow:0 0 0 2px {VIOLET}" if chosen else ""
    md(f'<div class="gw-card" style="display:flex;align-items:center;gap:20px;padding:22px 26px;height:96px;box-sizing:border-box;{ring}">'
       f'<div style="flex:none;width:52px;height:52px;border-radius:15px;background:{colour};color:#fff;display:flex;align-items:center;justify-content:center;'
       f'font-weight:800;font-size:1.2rem">{n}</div><div style="flex:1"><div style="font-weight:800;font-size:1.2rem;letter-spacing:-.02em">{esc(name)}</div>'
       f'<div style="color:{MUTED};font-size:.95rem">{esc(sub)}</div></div>{pill(badge, *badge_style)}'
       f'<div style="font-size:1.4rem;color:{INK};margin-left:6px">›</div><span class="gw-hit gw-hit-row"></span></div>')


def path_screen():
    progress_bar(0)
    title("How do you want to bring AI in?", "Pick one. You can come back and try another.")
    s = st.session_state
    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        existing = s.mode_val == "A course I already run"
        row_card(1, "Build something new", "A new AI course where few others teach",
                 "MOST DIFFERENT", ("#F1EEFF", "#3A2A9E"), VIOLET, not existing)
        st.button("Build something new", key="path_new", on_click=choose_path, args=("A new course idea",))
        row_card(2, "Add AI to a course I run", "Add AI to the work your course already teaches",
                 "BIGGEST REACH", ("#E2F6F3", "#0A5F55"), TEAL, existing)
        st.button("Add AI to a course I run", key="path_existing", on_click=choose_path, args=("A course I already run",))
        md(f'<p style="text-align:center;font-size:.9rem;margin-top:8px">We guessed <strong>{"adding AI to a course you run" if existing else "a new course"}</strong> '
           'from what you wrote.</p>')
        st.button("← Back", key="path_back", on_click=go, args=("start",))


def choose_path(mode):
    st.session_state.update(mode_val=mode, mode=mode, screen="who")


def who_screen():
    progress_bar(1)
    s = st.session_state
    keep("sector")
    n_ai = g.sectors().set_index("sector").ai_jobs.get(s.sector_val)
    title("Who is it for?", f"Picked from your description. {int(n_ai):,} {esc(o.work_word(s.sector_val))} job ads ask for AI." if n_ai else
          "Picked from your description. Change it if it isn't right.")
    guess = s.learners_val
    options = list(dict.fromkeys(([guess[:1].upper() + guess[1:]] if guess != "working professionals" else []) + LEARNERS))[:4]
    _, mid, _ = st.columns([1, 5, 1])
    with mid:
        cols = st.columns(4)
        for i, name in enumerate(options):
            on = guess.lower() == name.lower()
            with cols[i]:
                md(f'<div class="gw-card" style="height:160px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;'
                   f'text-align:center;padding:14px;box-sizing:border-box;{"box-shadow:0 0 0 2px " + VIOLET + ",0 12px 30px rgba(91,63,224,.14)" if on else ""}">'
                   f'<div style="width:52px;height:52px;border-radius:15px;display:flex;align-items:center;justify-content:center;'
                   f'background:{VIOLET if on else "#F1EEFF"};color:{"#fff" if on else VIOLET}">{PERSON}</div>'
                   f'<div style="font-weight:700;line-height:1.25">{esc(name)}</div><span class="gw-hit gw-hit-tile"></span></div>')
                st.button(name, key=f"who_{i}", use_container_width=True, on_click=pick, kwargs={"learners": name})
        a, b = st.columns(2)
        custom = a.text_input("Someone else? Describe them", key="learners_custom", placeholder="e.g. new HR executives")
        b.selectbox("Industry" + (" · suggested from your words" if s.get("sector_guessed") else " · please check"), sector_names(), format_func=label, key="sector")
        _, c, _ = st.columns([1, 1, 1])
        c.button("Continue", type="primary", use_container_width=True, on_click=leave_who, args=(custom,))


def leave_who(custom):
    if custom.strip():
        st.session_state.learners_val = custom.strip()
    confirmed("depth")


def depth_screen():
    progress_bar(2)
    best = o.guess_depth(st.session_state.text_val)
    title("How deep should it go?", f"We suggest <strong>{DEPTH_TILES[best][0]}</strong> from your description. "
          "Job ads ask for using and integrating AI far more often than courses teach it.")
    _, mid, _ = st.columns([1, 3.4, 1])
    with mid:
        cols = st.columns(3)
        for col, (key, (name, sub, levels, height)) in zip(cols, DEPTH_TILES.items()):
            on = st.session_state.depth_val == key
            badge = pill("BEST FIT", "#fff" if on else INK, VIOLET if on else "#fff") if key == best else ""
            with col:
                md(f'<div style="height:240px;display:flex;align-items:flex-end"><div class="gw-card" style="width:100%;height:{height}px;padding:16px;box-sizing:border-box;'
                   f'display:flex;flex-direction:column;justify-content:space-between;background:{VIOLET if on else "#fff"};color:{"#fff" if on else INK}">'
                   f'<div style="display:flex;justify-content:space-between;align-items:center"><span style="font-size:.78rem;font-weight:700;opacity:.7">{levels}</span>{badge}</div>'
                   f'<div><div style="font-weight:800;font-size:1.25rem;letter-spacing:-.02em">{name}</div>'
                   f'<div style="font-size:.85rem;opacity:.8;line-height:1.35">{sub}</div></div></div><span class="gw-hit gw-hit-depth"></span></div>')
                st.button(name, key="depth_" + key, use_container_width=True, on_click=pick, kwargs={"depth": key})
        _, c, _ = st.columns([1, 1.6, 1])
        c.button("Write my learning objectives", type="primary", use_container_width=True, on_click=go, args=("result",))


# ---------------- 3. Result ----------------
OVERLAP_STYLE = {"New": ("#DCFCE7", "#14532D"), "Partly covered": ("#FEF3C7", "#713F12"), "Already taught": ("#FEE2E2", "#7F1D1D"), "Unknown": ("#F1F2F7", "#3B3F52")}
ADVICE_STYLE = {"Differentiate": ("#FFE9E5", "#8A2A1B"), "Adapt or partner": ("#F1EEFF", "#3A2A9E"), "Add AI": ("#E2F6F3", "#0A5F55"),
                "Check syllabus": ("#F1F2F7", "#3B3F52"), "Related": ("#F1F2F7", "#3B3F52")}
SKILL_MARK = {"outcomes": ("✓", "#14532D", "in outcomes"), "description": ("~", "#713F12", "described only"), "missing": ("✗", "#7F1D1D", "not found")}


def objectives_list(objs, overlap):
    rows = []
    for n, (x, ov) in enumerate(zip(objs, overlap), 1):
        where = {"New": "No similar AI course teaches this in its outcomes",
                 "Partly covered": f"Partly covered by “{ov['course']}”", "Already taught": f"Already taught by “{ov['course']}”",
                 "Unknown": "No published outcomes to compare"}[ov["status"]]
        quote = f'<div class="gw-quote">Their outcome: “{esc(ov["outcome"])}”</div>' if ov.get("outcome") else ""
        rows.append(f'<div class="gw-lo"><div class="gw-num">{n}</div><div style="flex:1"><div class="gw-lo-text">{esc(x["text"])}.</div>'
                    f'<div class="gw-why">{pill(x["tag"], *TAG_STYLE[x["tag"]])} {esc(x["why"])}.</div>'
                    f'<div class="gw-why">{pill(ov["status"].upper(), *OVERLAP_STYLE[ov["status"]])} {esc(where)}</div>{quote}</div></div>')
    md('<div class="gw-card" style="padding:28px 32px;margin-top:18px"><div class="gw-eyebrow">Your learning objectives</div>'
       '<div style="font-weight:800;font-size:1.5rem;letter-spacing:-.035em">By the end of this course, learners will be able to:</div>' + "".join(rows) + "</div>")


def hero(badge, headline, sub, ring_value, ring_label, ring_note, chips, percent=False):
    ring = max(0, min(100, ring_value or 0))
    chip_html = "".join(f'<span class="gw-chip"><span style="color:{"#22C55E" if ok else CORAL}">{"✓" if ok else "✗"}</span>{esc(t)}</span>' for t, ok in chips)
    value = (f"{ring_value}%" if percent else ring_value) if ring_value is not None else "?"
    md(f'<div style="background:{INK};color:#fff;border-radius:32px;padding:44px 48px;margin-top:22px;display:flex;gap:36px;align-items:center;flex-wrap:wrap">'
       f'<div style="flex:1 1 440px">{badge}<div style="font-size:3.2rem;font-weight:800;letter-spacing:-.055em;line-height:1.02;margin:18px 0 14px">{esc(headline)}</div>'
       f'<div style="color:#C9CBD6;font-size:1.12rem">{sub}</div><div style="margin-top:18px">{chip_html}</div></div>'
       f'<div style="flex:none;margin:auto;text-align:center"><div style="width:220px;height:220px;border-radius:50%;'
       f'background:conic-gradient({CORAL} 0 {ring}%, #2A2D44 {ring}% 100%);display:flex;align-items:center;justify-content:center">'
       f'<div style="width:176px;height:176px;border-radius:50%;background:{INK};display:flex;flex-direction:column;align-items:center;justify-content:center">'
       f'<div style="font-size:3.4rem;font-weight:800;letter-spacing:-.05em;line-height:1">{value}</div>'
       f'<div style="color:#B7BACB;font-size:.85rem;max-width:120px;margin-top:4px">{esc(ring_label)}</div></div></div>'
       f'<div style="color:#B7BACB;font-size:.88rem;margin-top:12px">{esc(ring_note)}</div></div></div>')


def bars_card(eyebrow, rows, note):
    """rows: (name, job share, course share, label)."""
    top = max([max(j, c) for _, j, c, _ in rows] + [0.01])
    body = "".join(f'<div style="margin-top:16px"><div style="display:flex;justify-content:space-between;font-weight:700"><span>{esc(n)}</span>'
                   f'<span style="color:{VIOLET};font-size:.85rem">{esc(lab)}</span></div>'
                   f'<div style="height:9px;border-radius:5px;background:{VIOLET};width:{max(2, 100 * j / top):.0f}%;margin-top:7px"></div>'
                   f'<div style="height:9px;border-radius:5px;background:{CORAL};width:{max(2, 100 * c / top):.0f}%;margin-top:4px"></div></div>' for n, j, c, lab in rows)
    md(f'<div class="gw-card" style="padding:26px 28px;height:100%;box-sizing:border-box"><div class="gw-eyebrow">{eyebrow}</div>{body}'
       f'<div style="font-size:.82rem;color:{MUTED};margin-top:18px">{note}</div></div>')


def depth_card(levels):
    lad = g.ladder().set_index("maturity")
    steps = "".join(f'<div style="flex:1;height:{50 + 24 * i}px;border-radius:12px;background:{VIOLET if i in levels else "#F1F2F7"};'
                    f'color:{"#fff" if i in levels else MUTED};font-size:.75rem;font-weight:800;text-align:center;padding-top:8px">L{i}</div>' for i in range(1, 6))
    top = max(levels) if levels else 3
    r = lad.loc[top]
    md(f'<div class="gw-card" style="padding:26px 28px;height:100%;box-sizing:border-box"><div class="gw-eyebrow">At what depth</div>'
       f'<div style="display:flex;gap:8px;align-items:flex-end;height:160px;margin-top:10px">{steps}</div>'
       f'<div style="margin-top:16px;font-size:.98rem;color:{INK}"><strong>{r["name"]} (L{top})</strong> is {r.job_share:.0%} of AI job ads but only '
       f'{r.course_share:.0%} of AI courses.</div></div>')


def coming_card(h, work):
    tasks = "".join(f'<div style="background:#FFF4E2;color:#7C3F00;border-radius:14px;padding:11px 14px;margin-top:8px;font-weight:700;font-size:.92rem">'
                    f'{esc(t["addon"])}</div>' for t in h["ahead_tasks"])
    latent = f'{h["latent_share"]:.0%}' if h["latent_share"] is not None else "–"
    md(f'<div class="gw-card" style="padding:26px 28px;height:100%;box-sizing:border-box"><div class="gw-eyebrow" style="color:#0D9488">What\'s coming</div>'
       f'<div style="display:flex;gap:12px;align-items:center;margin-top:8px"><span style="font-size:3rem;font-weight:800;letter-spacing:-.05em;color:{AMBER}">{latent}</span>'
       f'<span style="font-size:.92rem;color:{MUTED}">of {esc(work)} job ads describe tasks AI can already help with</span></div>'
       f'{tasks or "<div style=color:" + MUTED + ";font-size:.9rem>No task-level forecast for this industry.</div>"}'
       f'<div style="font-size:.8rem;color:{MUTED};margin-top:12px">A forward-looking signal, not current demand.</div></div>')


def courses_section(facts, route, mode):
    lab, why = route
    md(f'<div style="margin:40px 0 6px"><div class="gw-eyebrow">From the course database</div>'
       f'<div style="font-weight:800;font-size:2rem;letter-spacing:-.045em">{"Courses like yours" if mode == "existing" else "Courses already out there"}</div>'
       f'<p class="gw-help" style="margin:4px 0 14px">The closest of 38,630 SkillsFuture and university courses, with their published learning outcomes.</p>'
       f'<div class="gw-card" style="padding:16px 22px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">{pill("RECOMMENDED: " + lab.upper(), INK, "#fff")}'
       f'<span style="color:#2A2E40">{esc(why)}</span></div></div>')
    if not facts:
        st.info("No course in the database is close to this description.")
    for f in facts:
        meta = [f["provider"], f["source"]]
        if f["hours"] == f["hours"] and f["hours"]:
            meta.append(f"{f['hours']:.0f} h")
        if f["fee"] == f["fee"] and f["fee"]:
            meta.append(f"SGD {f['fee']:,.0f}")
        if f["attendance"] == f["attendance"] and f["attendance"]:
            meta.append(f"{f['attendance']:,.0f} attended")
        marks = " ".join(f'<span title="{SKILL_MARK[v][2]}" style="color:{SKILL_MARK[v][1]};font-weight:700;margin-right:12px">{SKILL_MARK[v][0]} {esc(o.PLAIN_SKILL.get(k, k))}</span>'
                         for k, v in f["skills"].items())
        ai = pill("USES AI", "#F1EEFF", "#3A2A9E") if f["has_ai"] else pill("NO AI YET", "#F1F2F7", "#3B3F52")
        md(f'<div class="gw-card" style="padding:20px 24px;margin-top:12px"><div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">'
           f'<div><div style="font-weight:800;font-size:1.1rem;letter-spacing:-.02em">{esc(f["title"])}</div>'
           f'<div class="gw-why" style="margin-top:3px">{esc(" · ".join(meta))}</div></div>'
           f'<div style="flex:none;display:flex;gap:6px">{ai}{pill(f["advice"].upper(), *ADVICE_STYLE[f["advice"]])}</div></div>'
           f'<div style="margin:12px 0 4px;font-size:.88rem">{marks}</div><div class="gw-why" style="color:#2A2E40">{esc(f["why"])}</div></div>')
        if f["outcomes"]:
            with st.expander(f"Their published learning outcomes ({len(f['outcomes'])})"):
                for u in f["outcomes"]:
                    st.markdown(f"- {u}")
                st.caption(f"Source record {f['record_id']}, as published in the catalogue.")
    st.caption("✓ named in its learning outcomes · ~ only in its description · ✗ not found. Matching is by text similarity and keywords: "
               "it does not show how deeply a course teaches something. Check the full syllabus before deciding.")


def rewrite_key(objs, text):
    return hash((tuple(x["text"] for x in objs), text))


def rewritten(objs, text):
    """Swap in an accepted AI rewrite (same order, same evidence) unless the user asked for the original."""
    rec = st.session_state.get("rewrites", {}).get(rewrite_key(objs, text))
    if not rec or rec["status"] != "accepted" or st.session_state.get("show_original"):
        return objs
    return [dict(x, text=new, template=x["text"]) for x, new in zip(objs, rec["objectives"])]


def rewrite_panel(objs, text, learners, work):
    original = [x.get("template", x["text"]) for x in objs]
    key = rewrite_key([{"text": t} for t in original], text)
    rec = st.session_state.get("rewrites", {}).get(key)
    if rec and rec["status"] == "accepted":
        a, b = st.columns([3, 1.3])
        a.caption("Reworded by AI in your course's style. Each objective was checked: same meaning, no new facts or numbers.")
        b.toggle("Show original", key="show_original")
        return
    if rec:
        st.warning("The AI rewrite didn't pass the checks, so the original wording is kept.")
    with st.expander("✦ Make it sound like your course (optional AI rewrite)"):
        if not agent_available():
            st.caption("Connect a DeepSeek key to have AI reword these objectives in your course's style. The evidence stays the same.")
            st.button("Open AI key settings", key="rw_settings", on_click=go, args=("settings",), kwargs={"back_to": "result"})
            return
        st.caption("AI rewords the objectives using your description as a style guide. It cannot add facts or numbers: a checker rejects any change in meaning.")
        ok = st.checkbox("OK to use one paid request (under USD 0.01) from the USD 2 testing allowance", key="rw_ok")
        if st.button("Rewrite in my course's style", disabled=not ok, key="rw_go"):
            from agentic.transport import DeepSeekProvider
            from gapwise_rewrite import rewrite
            with st.spinner("Rewording and checking each objective…"):
                rec = rewrite([t.rstrip(".") for t in original], text, learners, work, DeepSeekProvider(2, max_tokens=1200),
                              anchors=[x.get("anchors", []) for x in objs])
            st.session_state.setdefault("rewrites", {})[key] = rec
            st.session_state.pop("show_original", None)
            st.rerun()


@st.cache_data(show_spinner=False)
def cached_facts(text, skills, mode, sector):
    return g.course_facts(text, list(skills), mode, sector=sector)


def result_screen():
    s = st.session_state
    sector, learners, depth, text = s.sector_val, s.learners_val.strip(), s.depth_val, s.text_val.strip()
    work = o.work_word(sector)
    h = g.horizons(sector)
    existing = s.mode_val == "A course I already run"
    with st.spinner("Reading job ads, running the checks and comparing 38,630 courses… (about 30 seconds the first time)"):
        if existing:
            matches = match_tasks(text, "")
            demand = g.task_demand_map(sector)
            objs = o.for_existing_course(sector, depth, matches, demand)
            skills = ("Generative AI", "LLM", "Responsible AI")
        else:
            lmin, lmax = o.DEPTH_LEVELS[depth]
            v = cached_verdict(sector, learners, text[:1200], lmin, lmax)
            p = g.new_path(sector)
            objs = o.for_new_course(sector, depth, v, p["open"], p["counts"], p["ai_courses"], h["ahead_tasks"])
            skills = tuple(dict.fromkeys([*v["skills"], *p["open"]]))
        facts = cached_facts(text, skills, "existing" if existing else "new", sector)
    objs = rewritten(objs, text)
    # The generic baseline is judged on wording only; its key ideas appear in almost any AI course.
    overlap = g.objective_overlap([x["text"] for x in objs], facts, [[] if x["tag"] == o.ALL else x.get("anchors", []) for x in objs])
    new_count = sum(ov["status"] == "New" for ov in overlap)
    depth_name = DEPTH_TILES[depth][0]
    learners = learners[:1].upper() + learners[1:]
    md(f'<div style="text-align:right;color:{MUTED};font-size:.9rem;margin-top:6px">{esc(label(sector))} · {esc(learners)} · {depth_name}</div>')
    if existing:
        tot = g.infusion_totals(sector)
        pct = round(100 * tot["with_ai"] / tot["courses_after_merging_variants"]) if tot["courses_after_merging_variants"] else None
        no_ai = sum(not f["has_ai"] for f in facts)
        hero(pill("ADD AI TO YOUR COURSE", "#5EEAD4", "#052E2B"), "Your course teaches work that AI is changing." if len(objs) > 1 else "Start with AI literacy.",
             f"<strong style='color:#fff'>{len(objs)} learning objectives</strong> to add. "
             + (f"{new_count} of them aren't taught with AI by any similar course, " if new_count else "Similar AI courses partly cover them, so see how yours can differ below, ")
             + f"and {no_ai} of the {len(facts)} closest courses have no AI yet.",
             pct, f"of {work} courses use AI", f"{tot['courses_after_merging_variants']:,} {work} courses checked", [], percent=True)
        md('<div style="height:14px"></div>')
        a, b, c = st.columns(3)
        with a:
            rows = [(TASKS[m["task"]]["label"], *demand.get(m["task"], (0, 0))) for m in matches[:3]]
            bars_card("What your course teaches", [(n, js, js * ai, f"{js:.0%}") for n, js, ai in rows] or [("No workplace task found", 0, 0, "")],
                      f"Purple: share of {esc(work)} job ads describing the task. Coral: those that also ask for AI.")
        with b:
            lmin, lmax = o.DEPTH_LEVELS[depth]
            depth_card(list(range(lmin, lmax + 1)))
        with c:
            coming_card(h, work)
        title_text = "Learning objectives to add AI to your course"
    else:
        per100 = v["courses_per_100"]
        concept = o.CONCEPTS.get(v.get("key"), v.get("concept") or "Your AI course")
        lmin, lmax = o.DEPTH_LEVELS[depth]
        hero(pill("✓ YES, BUILD IT", "#22C55E", "#052E14") if v["go"] else pill("NOT YET", "#FDE68A", "#713F12"),
             "Employers want it. Few courses teach it." if v["go"] else "The evidence isn't strong enough yet.",
             f"<strong style='color:#fff'>{esc(concept)}</strong> for {esc(learners)}, at the <strong style='color:#fff'>{depth_name}</strong> level. "
             + (f"{new_count} of your {len(objs)} objectives aren't taught by any similar course." if new_count else "Similar AI courses partly cover these objectives; see how to differ below."),
             round(per100) if per100 is not None else None, "AI courses for every 100 AI job ads",
             f"{v['sector_ai_jobs'] or 0:,} {work} job ads ask for AI", [(CHECK_NAMES.get(c["label"], c["label"]), c["passed"]) for c in v["checks"]])
        if v.get("alternative"):
            st.caption(f"Matched to your description. Also strong in {work}: {o.CONCEPTS.get(v['alternative_key'], v['alternative'])}.")
        a, b, c = st.columns(3)
        with a:
            rows = [(o.PLAIN_SKILL.get(e["skill"], e["skill"]), e["sector_job_mentions"] / max(e["sector_job_denominator"], 1),
                     e["pooled_course_mentions"] / max(e["pooled_course_denominator"], 1)) for e in v["skill_evidence"]]
            bars_card("What to teach", [(n, j, c_, f"{j:.0%} vs {c_:.0%}") for n, j, c_ in rows] or [("No skill evidence", 0, 0, "")],
                      f"Purple: share of {esc(work)} AI job ads naming the skill. Coral: share of all AI courses.")
        with b:
            depth_card(list(range(lmin, lmax + 1)))
        with c:
            coming_card(h, work)
        title_text = concept
    objectives_list(objs, overlap)
    rewrite_panel(objs, text, learners, work)
    plain = o.plain_text(objs, title_text)
    with st.expander("Edit the wording"):
        plain = st.text_area("Learning objectives", plain, height=300, label_visibility="collapsed")
    a, b, _ = st.columns([1.3, 1, 2])
    a.download_button("Download objectives", plain, file_name="learning_objectives.txt", type="primary", use_container_width=True)
    b.button("Start again", use_container_width=True, on_click=restart)
    courses_section(facts, g.route(facts, "existing" if existing else "new"), "existing" if existing else "new")
    steps = ["Read job ads", "Compared 38,630 courses", "Looked ahead"] + ([] if existing else ["Ran 6 checks"]) + ["Matched published outcomes"]
    md('<div class="gw-card" style="padding:18px 22px;margin-top:28px;display:flex;align-items:center;flex-wrap:wrap;gap:6px">'
       '<strong style="margin-right:8px">How Gapwise decided</strong>' + "".join(f'<span class="gw-step-chip">{x}</span>' for x in steps) + "</div>")
    if not existing:
        with st.expander("See the six checks"):
            for c in v["checks"]:
                st.markdown(f"{'✅' if c['passed'] else '❌'} **{CHECK_NAMES.get(c['label'], c['label'])}** ({c['label']}): {c['why']}")


# ---------------- Radar ----------------
ZONE_COLOUR = {"Build ahead": TEAL, "Hiring now": VIOLET, "Specialist": "#A5A8BC", "Quiet": "#C9CBD6"}


def radar_screen():
    r = g.radar().copy()
    r["name"] = r.sector.map(short)
    md('<div style="margin-top:30px"><div class="gw-eyebrow" style="color:#0D9488">Radar</div>'
       '<h1 style="font-size:3.4rem !important;max-width:760px;margin:4px 0 22px">Catch the next wave before it\'s in the job ads.</h1></div>')
    a, b = st.columns([1.75, 1], gap="medium")
    with a:
        md('<div class="gw-card" style="padding:18px 22px 4px"><div style="display:flex;justify-content:space-between;font-weight:700;font-size:.88rem">'
           f'<span style="color:#0D9488">● Build ahead</span><span style="color:{VIOLET}">● Hiring now</span></div></div>')
        x_max = float(r.ai_share.max()) * 1.1
        zone_box = alt.Chart(pd.DataFrame([{"x": 0, "x2": 0.05, "y": 0.35, "y2": float(r.latent_share.max()) * 1.05}])).mark_rect(
            color="#E2F6F3", cornerRadius=18).encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
        base = alt.Chart(r).encode(
            x=alt.X("ai_share:Q", title="Job ads asking for AI →", axis=alt.Axis(format="%", grid=False, titleColor=MUTED, labelColor=MUTED), scale=alt.Scale(domain=[0, x_max])),
            y=alt.Y("latent_share:Q", title="Jobs with tasks AI can already help with ↑", axis=alt.Axis(format="%", grid=False, titleColor=MUTED, labelColor=MUTED)))
        dots = base.mark_circle(opacity=.95, stroke="#fff", strokeWidth=2).encode(
            size=alt.Size("jobs:Q", scale=alt.Scale(range=[120, 1400]), legend=None),
            color=alt.Color("zone:N", scale=alt.Scale(domain=list(ZONE_COLOUR), range=list(ZONE_COLOUR.values())), legend=None),
            tooltip=[alt.Tooltip("sector:N", title="Industry"), alt.Tooltip("jobs:Q", format=","), alt.Tooltip("ai_share:Q", format=".1%", title="Ask for AI"),
                     alt.Tooltip("latent_share:Q", format=".0%", title="AI-ready tasks"), "zone:N"])
        names = base.transform_filter(((alt.datum.zone == "Build ahead") & (alt.datum.jobs >= 4000)) | (alt.datum.zone == "Hiring now") | (alt.datum.jobs >= 10000)).mark_text(align="left", dx=14, fontSize=12, fontWeight=600, color=INK).encode(text="name:N")
        st.altair_chart((zone_box + dots + names).properties(height=470).configure_view(strokeWidth=0).configure(background="#FFFFFF"), use_container_width=True)
        st.caption("Each bubble is an industry, sized by job ads. Build ahead: under 5% of ads ask for AI, but at least 35% describe tasks AI can "
                   "already help with. Declared display cut-offs; AI-ready tasks are a forward-looking signal, not demand.")
    with b:
        md('<span class="gw-dark"></span>')
        ahead = r[r.zone.eq("Build ahead")].sort_values("latent_share", ascending=False)
        options = ahead.sector.tolist() + [x for x in r.sort_values("jobs", ascending=False).sector if x not in set(ahead.sector)]
        sector = st.selectbox("Look closer at", options, format_func=label, key="radar_sector")
        row = r.set_index("sector").loc[sector]
        h = g.horizons(sector)
        colour = ZONE_COLOUR[row.zone]
        ring = round(100 * row.latent_share)
        ideas = "".join(f'<div style="background:#1A1D2E;border-radius:14px;padding:13px 16px;margin-top:8px;font-weight:700;color:#fff">{esc(t["addon"])}</div>'
                        for t in h["ahead_tasks"]) or '<div style="color:#B7BACB;font-size:.9rem;margin-top:8px">No task-level suggestion for this industry yet.</div>'
        md(f'<div style="margin-top:10px">{pill(row.zone, "#123B36" if row.zone == "Build ahead" else "#25204A", "#5EEAD4" if row.zone == "Build ahead" else "#B4A6FF")}</div>'
           f'<div style="color:#fff;font-size:2.1rem;font-weight:800;letter-spacing:-.045em;margin:10px 0 16px">{esc(short(sector))}</div>'
           f'<div style="display:flex;gap:18px;align-items:center"><div style="flex:none;width:112px;height:112px;border-radius:50%;'
           f'background:conic-gradient({colour} 0 {ring}%, #2A2D44 {ring}% 100%);display:flex;align-items:center;justify-content:center">'
           f'<div style="width:86px;height:86px;border-radius:50%;background:{INK};display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:1.6rem">{ring}%</div></div>'
           f'<div style="color:#C9CBD6;font-size:.95rem;line-height:1.5"><strong style="color:{colour}">{row.latent_share:.0%}</strong> of jobs have AI-ready tasks<br>'
           f'<strong style="color:#B4A6FF">{row.ai_share:.1%}</strong> ask for AI today<br>{int(row.jobs):,} job ads</div></div>'
           f'<div style="color:#8E92A8;font-size:.78rem;font-weight:800;letter-spacing:.06em;margin-top:22px">COURSES YOU COULD BUILD</div>{ideas}<div style="height:16px"></div>')
        st.button(f"Start a course for {short(sector)}", key="radar_go", on_click=from_radar, args=(sector, h))


def from_radar(sector, h):
    idea = h["ahead_tasks"][0]["addon"] if h["ahead_tasks"] else f"AI for {o.work_word(sector)} work"
    for k in ("mode", "text", "sector", "learners", "depth"):
        st.session_state.pop(k, None)
    st.session_state.update(mode_val="A new course idea", mode="A new course idea", text_val=f"A course on {idea[:1].lower() + idea[1:]} for {o.work_word(sector)} teams",
                            sector_val=sector, sector_guessed=True, learners_val="working professionals", depth_val="use", screen="path")


# ---------------- Settings ----------------
def connect():
    value = st.session_state.get("gw_key", "").strip()
    if not 12 <= len(value) <= 512:
        st.session_state.gw_key_error = "Enter a valid API key."
        return
    # Private credential file outside the project; never written to project outputs.
    fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w") as stream:
        if hasattr(os, "fchmod"):
            os.fchmod(stream.fileno(), 0o600)
        stream.write(value)
    st.session_state.pop("gw_key", None)
    st.session_state.pop("gw_key_error", None)


def disconnect():
    if KEY_PATH.exists():
        KEY_PATH.unlink()


def settings_screen():
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        md('<div style="margin-top:30px"><div class="gw-eyebrow">Settings</div><h1 style="font-size:3rem !important">AI key</h1>'
           '<p class="gw-help">You don\'t need a key. Gapwise writes learning objectives from the data without any AI service. '
           'With a DeepSeek key, it can also reword them in your course\'s style. You confirm each paid request.</p></div>')
        live = agent_available()
        spent = AgentBudget(2).spent()
        md(f'<div class="gw-card" style="padding:24px 26px;margin-bottom:14px"><div style="display:flex;justify-content:space-between;align-items:center">'
           f'<strong style="font-size:1.1rem">DeepSeek</strong>{pill("CONNECTED", "#DCFCE7", "#14532D") if live else pill("NOT CONNECTED", "#F1F2F7", "#3B3F52")}</div>'
           # "USD", not "$": Streamlit markdown renders $…$ as maths.
           f'<div style="margin-top:18px;font-size:.9rem;color:{MUTED}">Spending limit · USD {spent:.2f} of USD 2.00 used, across all sessions</div>'
           f'<div style="height:8px;border-radius:4px;background:#EEF0F5;margin-top:8px"><div style="height:8px;border-radius:4px;background:{VIOLET};width:{min(100, 50 * spent):.0f}%"></div></div></div>')
        # Hosted: the key is managed by the app owner in Streamlit Secrets; visitors cannot add or remove it.
        st.caption("This hosted demo uses a key set by the app owner. The AI rewrite stops when the spending limit is reached.")


def scroll_to_top():
    """Streamlit keeps the scroll position on rerun; a new screen must open at its top."""
    import streamlit.components.v1 as components
    if st.session_state.get("_last_screen") != st.session_state.screen:
        st.session_state._last_screen = st.session_state.screen
        components.html(f"<script>/*{st.session_state.screen}*/ const m = window.parent.document.querySelector('section.main') || window.parent.document.scrollingElement;"
                        " m.scrollTo({top: 0}); window.parent.scrollTo(0, 0);</script>", height=0)


def main():
    st.session_state.setdefault("screen", "start")
    md(CSS)
    header()
    {"start": start_screen, "path": path_screen, "ideas": ideas_screen, "who": who_screen, "depth": depth_screen, "result": result_screen,
     "radar": radar_screen, "settings": settings_screen}[st.session_state.screen]()
    scroll_to_top()


main()
