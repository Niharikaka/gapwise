"""Resumable DeepSeek JSON reviews with a durable, conservative spend ledger.

Only this module makes network requests. No keys are logged or saved in project files.
Unknown/time-out outcomes keep their maximum reservation: reruns cannot reset spend.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import urllib.request
import urllib.error

from data import ROOT, OUTPUT, load_jobs, load_courses, load_human
from taxonomy import SYSTEM_PROMPT, VERSION, text_hash, validate_review

# Peak, cache-miss prices verified 2026-10-22 at the official pricing page.
# Overestimate input tokens by UTF-8 bytes; reserve the full output cap per request.
PRICES = {"deepseek-flash": (0.30, 1.20), "deepseek-v4-pro": (1.32, 3.96)}
MAX_OUTPUT = 1200
LEDGER = OUTPUT / "api_ledger.jsonl"
CACHE = OUTPUT / "deepseek_reviews.jsonl"


# Private key file outside the project (owner-only); set ANL588_KEY_PATH to move it.
KEY_PATH = Path(os.environ.get("ANL588_KEY_PATH", Path(tempfile.gettempdir()) / "anl588_deepseek.key"))


def credential():
    value = os.environ.get("DEEPSEEK_API_KEY")
    if not value:
        value = KEY_PATH.read_text().strip() if KEY_PATH.exists() else ""
    if not value:
        raise RuntimeError("Set DEEPSEEK_API_KEY locally; never commit or print it")
    return value


def request_json(path, payload=None):
    req = urllib.request.Request("https://api.deepseek.com" + path,
        data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + credential(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=100) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # Do not print response bodies or headers: they may expose credential material.
        raise RuntimeError(f"DeepSeek HTTP {exc.code}") from None


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def cached_reviews():
    if not CACHE.exists():
        return {}
    return {r["cache_key"]: r for line in CACHE.read_text().splitlines() if (r := json.loads(line))}


class Budget:
    def __init__(self, maximum):
        self.maximum = maximum
        self.lock = threading.Lock()
        self.events = [json.loads(x) for x in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
        self.spent = sum(x["amount_usd"] for x in self.events)

    def reserve(self, key, maximum):
        with self.lock:
            if self.spent + maximum > self.maximum:
                return False
            event = {"event": "reserve", "key": key, "amount_usd": maximum,
                     "at": datetime.now(timezone.utc).isoformat()}
            append(LEDGER, event)
            self.spent += maximum
            return True

    def settle(self, key, reserved, actual):
        with self.lock:
            append(LEDGER, {"event": "settle", "key": key, "amount_usd": actual-reserved,
                            "actual_usd": actual})
            self.spent += actual-reserved


def review_one(record, model, budget, cache, output_lock):
    text = str(record["review_text"])
    key = text_hash(VERSION + SYSTEM_PROMPT + model + text)
    if key in cache:
        return "cached"
    # No truncation: missing evidence at the end must not become a false negative.
    user = json.dumps({"kind": record["kind"], "text": text}, ensure_ascii=False)
    input_bound = len((SYSTEM_PROMPT + user).encode()) + 256
    inp, out = PRICES[model]
    reserved = (input_bound*inp + MAX_OUTPUT*out)/1e6
    if not budget.reserve(key, reserved):
        return "budget_stopped"
    try:
        result = request_json("/chat/completions", {
            "model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"},
            "max_tokens": MAX_OUTPUT, "temperature": 0})
        usage = result["usage"]
        # Charge all input at cache-miss/peak rates to remain conservative.
        actual = (usage["prompt_tokens"]*inp + usage["completion_tokens"]*out)/1e6
        budget.settle(key, reserved, actual)
        choice = result["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Incomplete model output")
        repaired = False
        try:
            parsed = validate_review(json.loads(choice["message"]["content"]), text)
        except (ValueError, TypeError) as invalid:
            with output_lock:
                append(OUTPUT / "rejected_reviews.jsonl", {"record_id": record["record_id"],
                       "text_hash": record["text_hash"], "error": str(invalid),
                       "output": choice["message"]["content"]})
            feedback = "Your previous JSON failed validation: " + str(invalid) + ". Return a corrected JSON object. Copy quotations exactly, without ellipses or added punctuation. For ambiguous/no_evidence set maturity to 0 and maturity_quote to an empty string. Remove any skill or adoption opportunity that lacks a verbatim quote."
            repair_reserve = reserved + (len(choice["message"]["content"].encode()) + len(feedback.encode()))*inp/1e6
            if not budget.reserve(key+":repair", repair_reserve):
                return "budget_stopped"
            repair = request_json("/chat/completions", {
                "model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user},
                  {"role": "assistant", "content": choice["message"]["content"]},
                  {"role": "user", "content": feedback}],
                "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"},
                "max_tokens": MAX_OUTPUT, "temperature": 0})
            ru = repair["usage"]
            repair_cost = (ru["prompt_tokens"]*inp + ru["completion_tokens"]*out)/1e6
            budget.settle(key+":repair", repair_reserve, repair_cost)
            actual += repair_cost
            if repair["choices"][0]["finish_reason"] != "stop":
                raise ValueError("Incomplete repaired response")
            parsed = validate_review(json.loads(repair["choices"][0]["message"]["content"]), text)
            repaired = True
        row = {"cache_key": key, "record_id": record["record_id"], "kind": record["kind"],
               "text_hash": record["text_hash"], "source": "deepseek", "model_requested": model,
               "model_returned": result["model"], "taxonomy_version": VERSION,
               "prompt_hash": text_hash(SYSTEM_PROMPT), "usage": usage,
               "cost_upper_usd": actual, "repaired": repaired, "review": parsed}
        with output_lock:
            append(CACHE, row)
            cache[key] = row
        return "reviewed"
    except (ValueError, KeyError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
        with output_lock:
            append(OUTPUT / "api_errors.jsonl", {"record_id": record["record_id"],
                   "error_type": type(exc).__name__, "message": str(exc)[:150]})
        return "error"


def pilot_records(jobs, train_count=500, course_count=80):
    human = load_human()
    benchmark = jobs[jobs.record_id.isin(human.job_id)].copy()
    reserved_hashes = set(benchmark.text_hash)
    pool = jobs[~jobs.text_hash.isin(reserved_hashes)].copy()
    pool["old_positive"] = pool["AI Term Count"].gt(0)
    # Independent training examples, balanced on OLD screening only. No human test labels shown.
    sampled = pool.groupby("old_positive", group_keys=False).sample(n=train_count//2, random_state=588)
    job_sample = __import__("pandas").concat([benchmark, sampled]).drop_duplicates("text_hash")
    job_sample["kind"] = "job"
    cet, uni = load_courses()
    courses = __import__("pandas").concat([cet, uni], ignore_index=True)
    courses["old_positive"] = courses["AI Term Count"].gt(0)
    courses = courses.groupby("old_positive", group_keys=False).sample(n=course_count//2, random_state=588)
    courses["kind"] = "course"
    return __import__("pandas").concat([job_sample, courses], ignore_index=True).to_dict("records")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["models", "pilot", "all-jobs"])
    p.add_argument("--model", default="deepseek-flash", choices=list(PRICES))
    p.add_argument("--budget-usd", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--train-count", type=int, default=500)
    args = p.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    # Cross-process lock prevents two runners overspending a shared ledger.
    import fcntl
    with (OUTPUT / ".api_run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "models":
            models = request_json("/models")
            print(json.dumps({"models": [m["id"] for m in models["data"]]}))
            return
        jobs = load_jobs()
        records = pilot_records(jobs, args.train_count) if args.command == "pilot" else jobs.assign(kind="job").drop_duplicates("text_hash").to_dict("records")
        budget = Budget(args.budget_usd)
        cache = cached_reviews()
        counts = {}
        output_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(review_one, r, args.model, budget, cache, output_lock) for r in records]
            for i, future in enumerate(as_completed(futures), 1):
                outcome = future.result()
                counts[outcome] = counts.get(outcome, 0)+1
                if i % 25 == 0:
                    print(json.dumps({"completed": i, "outcomes": counts,
                                      "conservative_spend_usd": round(budget.spent, 6)}), flush=True)
        summary = {"command": args.command, "model": args.model, "requested": len(records),
                   "outcomes": counts, "budget_usd": args.budget_usd,
                   "conservative_spend_usd": budget.spent}
        (OUTPUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
