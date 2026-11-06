"""Read-only access to canonical V3 sources and stable identities."""
from pathlib import Path
import re
import pandas as pd
from taxonomy import text_hash

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUTPUT = ROOT / "outputs"

# UTF-8 bytes that were decoded as Windows-1252 (e.g. "â€¢" for "•"). Undefined
# cp1252 bytes survive as raw C1 characters, so both forms map back to one byte.
_CONTINUATION = {chr(b) if b in (0x81, 0x8D, 0x8F, 0x90, 0x9D) else bytes([b]).decode("cp1252"): b for b in range(0x80, 0xC0)}
_CONT = "[" + re.escape("".join(_CONTINUATION)) + "]"
_MOJIBAKE = re.compile(f"[Â-ß]{_CONT}|[à-ï]{_CONT}{{2}}|[ð-ô]{_CONT}{{3}}")


# Some sources were also NFKD-normalised after the mojibake, which rewrote the final byte
# (e.g. "ª" -> "a", "™" -> "TM", "¨" -> " " + combining diaeresis). Only unambiguous forms.
_NFKD_LAST = {"a": 0xAA, "o": 0xBA, "TM": 0x99, "3⁄4": 0xBE, "1⁄2": 0xBD, "1⁄4": 0xBC,
              " ̈": 0xA8, " ̄": 0xAF, " ́": 0xB4, " ̧": 0xB8, "μ": 0xB5}
_NFKD_TAIL = re.compile("â(" + _CONT + ")(" + "|".join(re.escape(k) for k in sorted(_NFKD_LAST, key=len, reverse=True)) + ")")
_INVISIBLE = re.compile("[​‌‍⁠﻿]|â\u0081 ")


def _decode_run(match):
    run = match.group()
    try:
        return bytes([ord(run[0])] + [_CONTINUATION[c] for c in run[1:]]).decode("utf-8")
    except UnicodeDecodeError:
        return run


def _decode_nfkd_tail(match):
    try:
        return bytes([0xE2, _CONTINUATION[match.group(1)], _NFKD_LAST[match.group(2)]]).decode("utf-8")
    except UnicodeDecodeError:
        return match.group()


def repair_text(value):
    """Undo cp1252 mojibake and drop zero-width characters; other text is unchanged."""
    if not isinstance(value, str):
        return value
    value = _NFKD_TAIL.sub(_decode_nfkd_tail, _MOJIBAKE.sub(_decode_run, value))
    # Word "Symbol"/Wingdings bullets decode to private-use code points that render as boxes.
    value = re.sub("[-]", "•", _INVISIBLE.sub("", value))
    return value.replace(" ", " ").replace(" ", " ")


def load_jobs():
    jobs = pd.read_csv(PROJECT / "02_cleaned_data/mcf_jobs_cleaned_ai_tagged.csv", low_memory=False)
    if jobs["Job ID"].isna().any() or not jobs["Job ID"].is_unique:
        raise ValueError("Job identity must be complete and unique")
    jobs["record_id"] = jobs["Job ID"]
    # Excludes old tags, human labels, salary and inferred occupation attributes.
    jobs["review_text"] = "TITLE: " + jobs.Title.fillna("") + "\nSKILLS: " + jobs.Skills.fillna("") + "\nDESCRIPTION: " + jobs.Description.fillna("")
    jobs["text_hash"] = jobs.review_text.map(text_hash)
    jobs["group_hash"] = jobs.Description.fillna("").map(lambda t: text_hash(t) if len(t.strip())>=80 else "")
    jobs.loc[jobs.group_hash.eq(""), "group_hash"] = jobs.loc[jobs.group_hash.eq(""), "text_hash"]
    return jobs


def load_courses():
    cet = pd.read_csv(PROJECT / "02_cleaned_data/myskillsfuture_courses_clean.csv", low_memory=False)
    uni = pd.read_csv(PROJECT / "02_cleaned_data/university_courses_clean.csv", low_memory=False)
    cet["record_id"] = "CET:" + cet.coursereferencenumber.astype(str)
    uni["record_id"] = "UNI:" + uni.provider.astype(str) + ":" + uni.course_code.astype(str)
    for frame in (cet, uni):
        if not frame.record_id.is_unique:
            raise ValueError("Course identity is not unique")
        # Hash the source text as published so existing labels and review caches still join.
        source_text = frame.course_text.fillna("")
        for column in frame.columns[frame.dtypes.eq(object)]:
            frame[column] = frame[column].map(repair_text)
        frame["review_text"] = frame.course_text.fillna("")
        frame["text_hash"] = source_text.map(text_hash)
        frame["group_hash"] = frame.text_hash
    return cet, uni


def load_human():
    return pd.read_csv(PROJECT / "05_analysis/validation_evidence/validation_labels_400.csv")
