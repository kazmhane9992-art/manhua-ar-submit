# -*- coding: utf-8 -*-
"""
المنطق الأساسي لمنصة مراجعة الترجمة
دوال معالجة النصوص والمطابقة والحفظ، مستقلة عن الواجهة.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CHAPTERS_EN = PROJECT_ROOT / "chapters" / "en"
CHAPTERS_KO = PROJECT_ROOT / "chapters" / "ko"
OUTPUT_AR = PROJECT_ROOT / "output" / "ar"
TRAINING_DIR = PROJECT_ROOT / "training_data"

LANG_LABELS = {"en": "الإنجليزية", "ko": "الكورية"}

PAGE_MARKER_RE = re.compile(r"^\s*---\s*(?:الصفحة|صفحة|Page)\s*\d+\s*---\s*$", re.IGNORECASE)
BUBBLE_PREFIX_RE = re.compile(r"^\s*(?:\[\]\s*:\s*|\(\)\s*:\s*|::|@|\$|#|//|م/)\s*")


def list_txt(directory: Path) -> list[str]:
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def read_lines(content: bytes) -> list[str]:
    return content.decode("utf-8-sig").splitlines()


def clean_lines(lines: list[str]) -> list[str]:
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if PAGE_MARKER_RE.match(s):
            continue
        cleaned.append(s)
    return cleaned


def strip_bubble(s: str) -> str:
    return BUBBLE_PREFIX_RE.sub("", s).strip()


def build_pairs(src_lines: list[str], tgt_lines: list[str]) -> list[dict]:
    src = clean_lines(src_lines)
    tgt = clean_lines(tgt_lines)
    n = min(len(src), len(tgt))
    rows = []
    for i in range(n):
        rows.append({"source": src[i], "translation": tgt[i], "include": True})
    for i in range(n, len(src)):
        rows.append({"source": src[i], "translation": "", "include": False})
    for i in range(n, len(tgt)):
        rows.append({"source": "", "translation": tgt[i], "include": False})
    return rows


def load_uploaded(src_uploads, tgt_uploads) -> list[dict]:
    targets = {Path(u.name).stem: u.getvalue() for u in tgt_uploads}
    rows: list[dict] = []
    used: set[str] = set()
    for u in src_uploads:
        stem = Path(u.name).stem
        src_lines = read_lines(u.getvalue())
        tgt_content = targets.get(stem)
        if tgt_content is not None:
            tgt_lines = read_lines(tgt_content)
            used.add(stem)
        else:
            tgt_lines = []
        for p in build_pairs(src_lines, tgt_lines):
            p["file"] = u.name
            rows.append(p)
    for u in tgt_uploads:
        if Path(u.name).stem in used:
            continue
        for p in build_pairs([], read_lines(u.getvalue())):
            p["file"] = u.name
            rows.append(p)
    return rows


def load_project(src_names: list[str], tgt_names: list[str], source_lang: str) -> list[dict]:
    src_dir = CHAPTERS_EN if source_lang == "en" else CHAPTERS_KO
    targets = {Path(n).stem: n for n in tgt_names}
    rows: list[dict] = []
    used: set[str] = set()
    for name in src_names:
        stem = Path(name).stem
        src_lines = read_lines((src_dir / name).read_bytes())
        tgt_name = targets.get(stem)
        if tgt_name is not None:
            tgt_lines = read_lines((OUTPUT_AR / tgt_name).read_bytes())
            used.add(stem)
        else:
            tgt_lines = []
        for p in build_pairs(src_lines, tgt_lines):
            p["file"] = name
            rows.append(p)
    for name in tgt_names:
        if Path(name).stem in used:
            continue
        for p in build_pairs([], read_lines((OUTPUT_AR / name).read_bytes())):
            p["file"] = name
            rows.append(p)
    return rows


def save_jsonl(rows: list[dict], filename: str, fmt: str, instruction: str, source_lang: str, strip_bubbles: bool) -> tuple[Path, int]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAINING_DIR / filename
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if not r.get("include", True):
                continue
            src = r.get("source", "").strip()
            tgt = r.get("translation", "").strip()
            if not src or not tgt:
                continue
            if strip_bubbles:
                src = strip_bubble(src)
                tgt = strip_bubble(tgt)
                if not src or not tgt:
                    continue
            if fmt == "alpaca":
                obj = {"instruction": instruction, "input": src, "output": tgt}
            else:
                obj = {"source_language": source_lang, "target_language": "ar", "source": src, "translation": tgt}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return path, count


def preview_jsonl(rows: list[dict], fmt: str, instruction: str, source_lang: str, strip_bubbles: bool, n: int = 3) -> list[str]:
    out = []
    for r in rows:
        if not r.get("include", True):
            continue
        src = r.get("source", "").strip()
        tgt = r.get("translation", "").strip()
        if not src or not tgt:
            continue
        if strip_bubbles:
            src = strip_bubble(src)
            tgt = strip_bubble(tgt)
        if fmt == "alpaca":
            obj = {"instruction": instruction, "input": src, "output": tgt}
        else:
            obj = {"source_language": source_lang, "target_language": "ar", "source": src, "translation": tgt}
        out.append(json.dumps(obj, ensure_ascii=False))
        if len(out) >= n:
            break
    return out


def count_submitted_files() -> int:
    """عدد ملفات JSONL المرفوعة للتدريب (يستثني الملف المدمج)."""
    if not TRAINING_DIR.exists():
        return 0
    return sum(
        1
        for p in TRAINING_DIR.glob("*.jsonl")
        if p.name != "merged_finetune.jsonl"
    )


def archive_submitted_files() -> list[str]:
    """نقل ملفات الرفع بعد معالجتها إلى مجلد archive/ حتى لا يُعاد تدريبها.

    يعيد قائمة أسماء الملفات المنقولة.
    """
    if not TRAINING_DIR.exists():
        return []
    archive_dir = TRAINING_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in sorted(TRAINING_DIR.glob("*.jsonl")):
        if p.name == "merged_finetune.jsonl":
            continue
        p.replace(archive_dir / p.name)
        moved.append(p.name)
    return moved
