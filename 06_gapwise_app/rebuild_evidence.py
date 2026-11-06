"""Rebuild every evidence table the Gapwise app reads, from the cleaned data. Offline: no API calls.

The AI labels reuse the cached DeepSeek reviews in outputs/deepseek_reviews.jsonl, so nothing is re-sent.
Run from this folder: python rebuild_evidence.py   (about 10 minutes; the shipped outputs/ already hold the results)
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Refined AI labels for jobs and courses", ["build_labels.py"]),
    ("Validation uncertainty", ["evaluate_uncertainty.py"]),
    ("Demand and supply metrics on refined labels", ["run_metrics.py", "--scenarios", "strict"]),
    ("Skill gaps, maturity and candidate scores", ["supplemental.py", "--scenarios", "strict"]),
    ("Task demand and AI suggestions for existing courses", ["course_infusion.py"]),
    ("Radar and per-course skill flags", ["gapwise_data.py", "build"]),
    ("Automated checks", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
]

for name, command in STEPS:
    print(f"\n=== {name} ===", flush=True)
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
print("\nEvidence rebuilt. Start the app with: python -m streamlit run gapwise_app.py")
