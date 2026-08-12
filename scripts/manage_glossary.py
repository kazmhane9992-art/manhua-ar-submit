#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""أداة إدارة قاموس الترجمة (glossary.json).

الاستخدام:
  python scripts/manage_glossary.py list
  python scripts/manage_glossary.py add --en "Solo Leveling" --ar "تسوية الكيان" --cat title --notes "..."
  python scripts/manage_glossary.py add --ko "레벨" --ar "المستوى" --cat skill
  python scripts/manage_glossary.py search --en Solo
  python scripts/manage_glossary.py search --ar مستوى
  python scripts/manage_glossary.py remove --en "Solo Leveling"
  python scripts/manage_glossary.py remove --id 3
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

GLOSSARY_PATH = os.path.join(os.path.dirname(__file__), "..", "glossary", "glossary.json")

CATEGORIES = {"person", "place", "skill", "object", "title", "sfx", "organization", "other"}


def load():
    if not os.path.exists(GLOSSARY_PATH):
        return {"entries": []}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save(data):
    with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def clean(value):
    if value is None:
        return ""
    value = value.strip()
    return value


def show(entry, index=None):
    prefix = f"[{index}] " if index is not None else ""
    parts = []
    if entry.get("term_en"):
        parts.append(f"EN: {entry['term_en']}")
    if entry.get("term_ko"):
        parts.append(f"KO: {entry['term_ko']}")
    parts.append(f"AR: {entry.get('translation_ar', '')}")
    parts.append(f"cat: {entry.get('category', '')}")
    notes = entry.get("notes", "")
    if notes:
        parts.append(f"notes: {notes}")
    print(prefix + " | ".join(parts))


def cmd_list(args):
    data = load()
    entries = data["entries"]
    if not entries:
        print("القاموس فارغ.")
        return
    for i, entry in enumerate(entries, 1):
        show(entry, i)


def cmd_add(args):
    data = load()
    entry = {
        "term_en": clean(args.en),
        "term_ko": clean(args.ko),
        "translation_ar": clean(args.ar),
        "category": clean(args.cat) if args.cat else "other",
        "notes": clean(args.notes) if args.notes else "",
    }
    if not entry["translation_ar"]:
        print("خطأ: الترجمة العربية (--ar) مطلوبة.")
        sys.exit(1)
    if args.cat and args.cat not in CATEGORIES:
        print(f"تحذير: التصنيف '{args.cat}' غير معروف. المقبول: {sorted(CATEGORIES)}")

    term = entry["term_en"] or entry["term_ko"]
    for existing in data["entries"]:
        if term and (existing.get("term_en") == entry["term_en"]
                     or existing.get("term_ko") == entry["term_ko"]):
            if term:
                print(f"موجود بالفعل: هذا المصطلح معرّف مسبقاً.")
                show(existing)
                sys.exit(1)

    data["entries"].append(entry)
    save(data)
    print("تمت الإضافة:")
    show(entry)


def cmd_search(args):
    data = load()
    needle = clean(args.en) or clean(args.ko) or clean(args.ar)
    if not needle:
        print("حدد --en أو --ko أو --ar للبحث.")
        sys.exit(1)
    found = False
    for i, entry in enumerate(data["entries"], 1):
        haystack = " ".join([
            entry.get("term_en", ""), entry.get("term_ko", ""),
            entry.get("translation_ar", ""), entry.get("notes", "")
        ])
        if needle in haystack:
            show(entry, i)
            found = True
    if not found:
        print("لا توجد نتائج.")


def cmd_remove(args):
    data = load()
    before = len(data["entries"])
    remaining = []
    removed = []
    for entry in data["entries"]:
        if args.id is not None and data["entries"].index(entry) + 1 == args.id:
            removed.append(entry)
            continue
        term_en = clean(args.en) if args.en else ""
        term_ko = clean(args.ko) if args.ko else ""
        if (term_en and entry.get("term_en") == term_en) or (term_ko and entry.get("term_ko") == term_ko):
            removed.append(entry)
            continue
        remaining.append(entry)
    if not removed:
        print("لم يُعثر على مصطلح مطابق.")
        sys.exit(1)
    for entry in removed:
        show(entry)
        print("  ^ تمت إزالته.")
    data["entries"] = remaining
    save(data)
    print(f"أزيل {len(removed)} من أصل {before}.")


def main():
    parser = argparse.ArgumentParser(description="إدارة قاموس الترجمة")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="عرض كل المصطلحات")
    p_list.set_defaults(func=cmd_list)

    p_add = sub.add_parser("add", help="إضافة مصطلح")
    p_add.add_argument("--en", help="المصطلح بالإنجليزية")
    p_add.add_argument("--ko", help="المصطلح بالكورية")
    p_add.add_argument("--ar", required=True, help="الترجمة العربية المعتمدة")
    p_add.add_argument("--cat", help=f"التصنيف ({', '.join(sorted(CATEGORIES))})")
    p_add.add_argument("--notes", help="ملاحظات السياق والاستخدام")
    p_add.set_defaults(func=cmd_add)

    p_search = sub.add_parser("search", help="البحث عن مصطلح")
    p_search.add_argument("--en", help="بحث بالإنجليزية")
    p_search.add_argument("--ko", help="بحث بالكورية")
    p_search.add_argument("--ar", help="بحث بالعربية")
    p_search.set_defaults(func=cmd_search)

    p_remove = sub.add_parser("remove", help="حذف مصطلح")
    p_remove.add_argument("--en", help="المصطلح بالإنجليزية")
    p_remove.add_argument("--ko", help="المصطلح بالكورية")
    p_remove.add_argument("--id", type=int, help="رقم المصطلح من list")
    p_remove.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
