"""Execute every original analytical block with versioned evidence-label scenarios.

The V3 generator is imported read-only. Adaptations are saved verbatim in an executed
notebook. All tables and figures are exported per scenario; legacy files are untouched.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/anl588_v4_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/anl588_v4_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/anl588_v4_numba")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import nbformat
import numpy as np
import pandas as pd

from data import ROOT, PROJECT, OUTPUT

SCENARIOS = ("legacy", "strict", "expanded", "student")


ADAPTER = '''
# V4 explicit scenario adapter. Canonical V3 CSV fields are retained on disk.
label_dir = Path(__LABEL_PATH__)
jl = pd.read_csv(label_dir / 'job_labels.csv').set_index('record_id')
cl = pd.read_csv(label_dir / 'course_labels.csv').set_index('record_id')
scenario = __SCENARIO__
jobs['legacy_ai'] = jobs['ai'].copy()
if scenario != 'legacy':
    field = 'strict_ai' if scenario == 'strict' else ('student_ai' if scenario == 'student' else 'expanded_ai')
    jobs['ai'] = jobs['Job ID'].map(jl[field])
    if jobs['ai'].isna().any():
        raise ValueError('Incomplete job-label coverage')
    jobs['ai'] = jobs['ai'].astype(bool)
    for frame, identity in [(msf, 'CET:' + msf.coursereferencenumber.astype(str)),
                            (uni, 'UNI:' + uni.provider.astype(str) + ':' + uni.course_code.astype(str))]:
        # Student is jobs-only; compare with expanded evidence screening on supply.
        frame['ai_match'] = identity.map(cl['strict_ai' if scenario == 'strict' else 'expanded_ai'])
        if frame['ai_match'].isna().any():
            raise ValueError('Incomplete course-label coverage')
        frame['ai_match'] = frame['ai_match'].astype(bool)
    # Preserve shared dictionary themes; newly found positives require a transparent
    # unassigned category rather than invented topic assignments.
    for frame, flag in [(jobs, 'ai'), (msf, 'ai_match'), (uni, 'ai_match')]:
        frame.loc[frame[flag] & frame['AI Categories'].isna(), 'AI Categories'] = 'AI evidence - theme unassigned'
uni_a = uni[(uni['provider'] != 'NUS') | uni['is_currently_offered'].eq(True)].copy()
print('V4 scenario:', scenario, 'AI jobs:', int(jobs.ai.sum()), 'CET:', int(msf.ai_match.sum()),
      'university analysis sample:', int(uni_a.ai_match.sum()))
'''


def strip_fixed_interpretations(source):
    """Remove legacy narrative claims that may not hold after relabelling."""
    tree = ast.parse(source)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
            args = node.value.args
            if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
                text = args[0].value
                if text.startswith(("Interpretation:", "Limitation:", "Caveat:", "(LLM", "courses —")):
                    spans.append((node.lineno, node.end_lineno))
    lines = source.splitlines()
    for first, last in sorted(spans, reverse=True):
        lines[first-1:last] = ["# Legacy fixed narrative removed; inspect the recomputed evidence tables."]
    return "\n".join(lines)


def adapted_cells(scenario):
    path = PROJECT / "05_analysis/build_company_x_eda.py"
    spec = importlib.util.spec_from_file_location("legacy_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cells = []
    for c in module.CELLS:
        if c.cell_type != "code":
            # Keep structure/methods, remove the hardcoded final course recommendation.
            if "### 🏆 Recommended course concept" in c.source:
                continue
            c = nbformat.v4.new_markdown_cell(c.source + "\n\n*V4: original analytical specification; numerical conclusions must come from this run's outputs.*")
            cells.append(c)
            continue
        source = strip_fixed_interpretations(c.source)
        for before, after in {
            'taught in AI courses': 'mentioned in course descriptions',
            'required in AI postings': 'mentioned in AI postings',
            'What employers ask for vs what courses teach': 'Skill mentions in job and course descriptions',
            'Right-pointing gaps (teal beyond clay) = skills employers want more than courses teach.': 'Text-mention gaps are proxies; descriptions may omit skills actually taught.',
        }.items():
            source = source.replace(before, after)
        if "def find_data_dir():" in source:
            # Pin to V3: original root fallback could silently select V2.
            a = source.index("def find_data_dir():")
            b = source.index("DATA_DIR = find_data_dir()")
            source = source[:a] + f"def find_data_dir():\n    return Path({str(PROJECT / '02_cleaned_data')!r})\n\n" + source[b:]
            source += "\n" + ADAPTER.replace("__LABEL_PATH__", repr(str(OUTPUT))).replace("__SCENARIO__", repr(scenario))
        if "MATURITY = {" in source:
            start = source.index("def maturity_level(text):")
            end = source.index("ai_jobs = jobs[jobs.ai].copy()", start)
            # Context-gated task classifier. Generic CI/CD/Kubernetes do not imply AI deployment.
            source = source[:start] + "from taxonomy import maturity_level\n" + source[end:]
            if scenario != "legacy":
                line = "ai_jobs['maturity'] = ai_jobs['jtext_lower'].map(maturity_level)"
                source = source.replace(line, line + "\nreview_mat = ai_jobs['Job ID'].map(jl['maturity'])\nai_jobs.loc[review_mat.gt(0), 'maturity'] = review_mat[review_mat.gt(0)].astype(int)")
        if "SKILLS = {" in source:
            source = source.replace("'MLOps': r'mlops|ci/cd|model serving'", "'MLOps': r'\\bmlops\\b|model serving|model monitoring'")
            old = "for k, rx in SK_RX.items():\n    jobs[f'sk__{k}'] = jobs['jtext_lower'].str.contains(rx)"
            new = '''
import hashlib
skill_cache = label_dir / 'skill_features.csv'
skill_meta = label_dir / 'skill_features_meta.json'
skill_signature = hashlib.sha256((json.dumps(SKILLS, sort_keys=True) + ''.join(jobs['Job ID'].map(jl.text_hash))).encode()).hexdigest()
if skill_cache.exists() and skill_meta.exists() and json.loads(skill_meta.read_text()).get('signature') == skill_signature:
    cached_skills = pd.read_csv(skill_cache).set_index('Job ID')
    if set(cached_skills.index) != set(jobs['Job ID']):
        raise ValueError('Skill feature cache identity mismatch')
    for k in SK_RX:
        jobs[f'sk__{k}'] = jobs['Job ID'].map(cached_skills[k]).astype(bool)
else:
    for k, rx in SK_RX.items():
        jobs[f'sk__{k}'] = jobs['jtext_lower'].str.contains(rx)
    cached_skills = jobs[['Job ID'] + [f'sk__{k}' for k in SK_RX]].copy()
    cached_skills.columns = ['Job ID'] + list(SK_RX)
    cached_skills.to_csv(skill_cache, index=False)
    skill_meta.write_text(json.dumps({'signature': skill_signature}))
'''
            source = source.replace(old, new)
        if "exp_years = pd.to_numeric" in source:
            # Unknown experience remains visible as missingness, not zero experience.
            source = source.replace(".fillna(0).clip(0, 40).rename('min_years_exp')", ".clip(0, 40).rename('min_years_exp')")
            source = source.replace("X2 = ", "X2 = ", 1)  # complete-case handling below, if named differently preserve source
            # statsmodels accepts finite design only; median + missing indicator preserves rows.
            marker = "emp_mlb = MultiLabelBinarizer()"
            source = source.replace(marker, "exp_missing = exp_years.isna().astype(float).rename('experience_missing')\nexp_years = exp_years.fillna(exp_years.median())\n" + marker)
            source = source.replace("[X.drop(columns='const'), exp_years, emp_df]", "[X.drop(columns='const'), exp_years, exp_missing, emp_df]")
        if "score = pd.DataFrame({" in source and "'salary_value':[" in source:
            # Export analyst hypotheses for comparison but do not disguise them as fitted scores.
            source += "\nscore['score_source'] = 'legacy analyst-assigned hypothesis; not used by V4 recommender'"
        cells.append(nbformat.v4.new_code_cell(source))
    return cells


def run(scenario):
    dest = OUTPUT / "metrics" / scenario
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "figures").mkdir(exist_ok=True)
    (dest / "tables").mkdir(exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(dest)
    namespace = {"__name__": "__main__"}
    cells = adapted_cells(scenario)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    manifest = {"scenario": scenario, "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running", "completed_cells": [], "tables": [], "figures": [],
                "source_generator_sha256": hashlib.sha256((PROJECT / "05_analysis/build_company_x_eda.py").read_bytes()).hexdigest(),
                "job_labels_sha256": hashlib.sha256((OUTPUT / "job_labels.csv").read_bytes()).hexdigest(),
                "course_labels_sha256": hashlib.sha256((OUTPUT / "course_labels.csv").read_bytes()).hexdigest(),
                "caveats": ["Strict/expanded combine sparse DeepSeek reviews with explicitly identified local screening elsewhere.",
                            "Student predictions are AI-supervised model inferences, not a full LLM review.",
                            "Legacy scenario retains dictionary flags but uses the same corrected metric implementation as other scenarios.",
                            "Course counts and text similarity do not establish teaching capacity or actual skill deficits."]}
    import matplotlib.pyplot as plt
    def save_figures(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            name = f"figure_{len(manifest['figures'])+1:03d}.png"
            fig.savefig(dest / "figures" / name, dpi=110, bbox_inches="tight")
            manifest["figures"].append(name)
        plt.close("all")
    plt.show = save_figures
    import io
    import contextlib
    execution_count = 0
    try:
        for index, cell in enumerate(cells):
            if cell.cell_type != "code":
                continue
            execution_count += 1
            cell.execution_count = execution_count
            captured = []
            def capture_display(*objects, **kwargs):
                for obj in objects:
                    if hasattr(obj, "data") and isinstance(obj.data, pd.DataFrame):
                        obj = obj.data
                    if isinstance(obj, (pd.DataFrame, pd.Series)):
                        captured.append(nbformat.v4.new_output("display_data", data={"text/plain": obj.to_string(max_rows=30)}))
            namespace["display"] = capture_display
            # The original setup imports display: restore immediately after the setup executes.
            stream = io.StringIO()
            started = time.monotonic()
            with contextlib.redirect_stdout(stream):
                exec(compile(cell.source, f"scenario_{scenario}_cell_{execution_count}", "exec"), namespace)
            namespace["display"] = capture_display
            if execution_count == 1:
                original_table_renderer = namespace["save_table_png"]
                def export_table(df, name, *args, **kwargs):
                    df.to_csv(dest / "tables" / f"{name}.csv", index=kwargs.get("index", False))
                    return original_table_renderer(df, name, *args, **kwargs)
                namespace["save_table_png"] = export_table
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue())] + captured
            manifest["completed_cells"].append({"cell": execution_count, "seconds": round(time.monotonic()-started, 2)})
            (dest / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"{scenario}: cell {execution_count} completed", flush=True)
        # Export analytical tables, not row-level full-text source copies.
        excluded = {"jobs", "msf", "uni", "uni_a", "monthly", "reg", "ai_j", "ai_jobs", "jl", "cl", "ud", "jex", "pos_ex", "all_ex", "cat_ex", "corpus"}
        for name, value in list(namespace.items()):
            if name in excluded or name.startswith("_"):
                continue
            if isinstance(value, pd.DataFrame) and len(value) <= 100000:
                if any(c in value.columns for c in ["job_text", "review_text", "Description"]):
                    continue
                value.to_csv(dest / "tables" / f"{name}.csv", index=True)
                manifest["tables"].append(name)
        overall = {"scenario": scenario, "jobs": len(namespace["jobs"]),
                   "ai_jobs": int(namespace["jobs"].ai.sum()),
                   "ai_rate": float(namespace["jobs"].ai.mean()),
                   "cet_ai_courses": int(namespace["msf"].ai_match.sum()),
                   "university_ai_courses": int(namespace["uni_a"].ai_match.sum())}
        for name in ["beta", "salary_cutoff"]:
            if name in namespace:
                overall[name] = float(namespace[name])
        salary_models = []
        for name in ["ols", "ols_trim", "ols_ext"]:
            model = namespace.get(name)
            if model is not None and 'AI' in model.params:
                ci = model.conf_int().loc['AI']
                salary_models.append({"specification": name, "n": int(model.nobs),
                                      "association_pct": float(100*np.expm1(model.params['AI'])),
                                      "ci_low_pct": float(100*np.expm1(ci.iloc[0])),
                                      "ci_high_pct": float(100*np.expm1(ci.iloc[1])),
                                      "p_value": float(model.pvalues['AI'])})
        pd.DataFrame(salary_models).to_csv(dest / "tables/salary_models.csv", index=False)
        (dest / "summary.json").write_text(json.dumps(overall, indent=2))
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        cell.outputs = [nbformat.v4.new_output("error", ename=type(exc).__name__, evalue=str(exc), traceback=traceback.format_exc().splitlines())]
        raise
    finally:
        nbformat.write(notebook, dest / "analysis_executed.ipynb")
        (dest / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        if manifest['status'] == 'complete':
            from embed_figures import embed
            embed(dest)
        os.chdir(original_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["legacy", "strict", "expanded"])
    args = parser.parse_args()
    for scenario in args.scenarios:
        run(scenario)
