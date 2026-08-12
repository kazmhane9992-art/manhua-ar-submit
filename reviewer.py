#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""المراجعة التلقائية للترجمة عبر OpenAI API.

يأخذ كل زوج (نص أصلي + ترجمة عربية) ويرسله إلى نموذج لغوي يراجعه:
  - يقبل الترجمة كما هي إذا كانت سليمة، أو
  - يقترح تصحيحاً في الصياغة مع الحفاظ على المعنى.

يستفيد النموذج من قاموس المصطلحات (glossary.json) لضمان الاتساق.

مفتاح API: من متغير البيئة OPENAI_API_KEY
  set OPENAI_API_KEY=sk-...
"""

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
GLOSSARY_PATH = PROJECT_ROOT / "glossary" / "glossary.json"

LANG_LABELS = {"en": "الإنجليزية", "ko": "الكورية"}
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = (
    "أنت مترجم محترف ومدقق ترجمة متخصص في المانجا والمانهوا. "
    "مهمتك مراجعة ترجمة عربية مقابل نصها الأصلي (إنجليزي أو كوري).\n"
    "قواعدك:\n"
    "1. لا ترجمة حرفية؛ الهدف سلاسة المعنى وروح النص.\n"
    "2. استخدم المصطلحات الواردة في القاموس المرفق حرفياً.\n"
    "3. حافظ على علامات الترقيم ورموز الفقاعات وأسماء الشخصيات كما هي.\n"
    "4. إذا كانت الترجمة سليمة فلا تغيّرها أبداً.\n"
    "5. إن كانت الصياغة ضعيفة أو مختلة أو مفقودة المعنى، أعد كتابتها بأسلوب طبيعي.\n"
    "6. لا تختلق جملة جديدة؛ عبّر عن معنى النص الأصلي نفسه."
)


def load_glossary() -> list[dict]:
    """قراءة قاموس المصطلحات."""
    if not GLOSSARY_PATH.exists():
        return []
    try:
        with GLOSSARY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", [])
    except Exception:
        return []


def glossary_to_text(entries: list[dict], limit: int = 400) -> str:
    """تحويل القاموس إلى نص مختصر يُمرَّر للنموذج."""
    lines = []
    for e in entries:
        term = e.get("term_en") or e.get("term_ko") or ""
        ar = e.get("translation_ar") or ""
        if term and ar:
            lines.append(f"- {term} = {ar}")
    if len(lines) > limit:
        lines = lines[:limit]
        lines.append("...(باقي المصطلحات غير المعروضة لها ترجمات اعتيادية)")
    return "\n".join(lines)


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("مكتبة openai غير مثبتة. ثبّتها:  pip install openai")
        sys.exit(1)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print(
            "خطأ: مفتاح OPENAI_API_KEY غير مضبوط.\n"
            "  set OPENAI_API_KEY=sk-...  (في الطرفية)\n"
            "أو أضفه في Secrets عند النشر على Hugging Face."
        )
        sys.exit(1)
    return OpenAI(api_key=key)


def _build_messages(batch: list[dict], source_lang: str, glossary_text: str, extra_notes: str) -> list[dict]:
    pairs_json = json.dumps(
        [{"i": idx, "source": p["source"], "translation": p["translation"]} for idx, p in enumerate(batch)],
        ensure_ascii=False,
    )
    user_content = (
        f"لغة النص الأصلي: {LANG_LABELS.get(source_lang, source_lang)}\n\n"
        f"القاموس:\n{glossary_text}\n\n"
        f"ملاحظات إضافية:\n{extra_notes or 'لا يوجد.'}\n\n"
        f"الجمل للمراجعة (تنسيق JSON):\n{pairs_json}\n\n"
        "أعد JSON بالشكل التالي حصراً (مصفوفة، عنصر واحد لكل جملة، مع الإبقاء على المفتاح i نفسه):\n"
        '[{"i": 0, "status": "accepted" | "corrected", "translation": "الترجمة النهائية", "reason": "سبب التصحيح إن وُجد"}]'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_json(text: str) -> list | None:
    """استخراج قائمة JSON من رد النموذج (يدعم وجود نص حولها)."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # البحث عن أول [ ... ] في النص
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None


def _review_batch(client, batch: list[dict], source_lang: str, glossary_text: str, extra_notes: str, model: str) -> list[dict]:
    """مراجعة دفعة واحدة من الجمل. يعيد قائمة بنفس ترتيب الدفعة.

    عند فشل استدعاء النموذج (انقطاع، نقص رصيد، خطأ API...) تُقبل الترجمات
    كما هي دون تعطيل سير الرفع.
    """
    result_map = {p["i"]: p for p in batch}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=_build_messages(batch, source_lang, glossary_text, extra_notes),
            temperature=0.2,
        )
    except Exception as e:
        print(f"[reviewer] فشل استدعاء النموذج ({model}): {e}")
        print("[reviewer] تُقبل الترجمات كما هي دون تصحيح.")
        return [{"i": p["i"], "status": "accepted", "translation": p["translation"], "reason": ""} for p in batch]
    parsed = _extract_json(response.choices[0].message.content)
    if parsed is None:
        # فشل تحليل الرد: اعتمد الترجمات كما هي
        return [{"i": p["i"], "status": "accepted", "translation": p["translation"], "reason": ""} for p in batch]
    out = []
    for item in parsed:
        i = item.get("i")
        if i in result_map:
            status = item.get("status", "accepted")
            translation = (item.get("translation") or result_map[i]["translation"]).strip()
            out.append({
                "i": i,
                "status": "corrected" if status == "corrected" else "accepted",
                "translation": translation or result_map[i]["translation"],
                "reason": item.get("reason", "").strip(),
            })
    # التأكد من تغطية كل الجمل (لأي جملة لم يعُد بها النموذج)
    covered = {o["i"] for o in out}
    for p in batch:
        if p["i"] not in covered:
            out.append({"i": p["i"], "status": "accepted", "translation": p["translation"], "reason": ""})
    return sorted(out, key=lambda x: x["i"])


def review_pairs(
    rows: list[dict],
    source_lang: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = 15,
    extra_notes: str = "",
) -> list[dict]:
    """مراجعة كل الجمل المرفوعة.

    rows: قائمة قواميس تحتوي source وtranslation (مثل مخرجات load_uploaded).
    يعيد قائمة بنفس بنية rows لكن مع source/translation نهائية +
      status وreason لكل جملة (ملائمة للعرض والحفظ).
    """
    client = get_client()
    glossary_text = glossary_to_text(load_glossary())

    # الجمل المكتملة فقط (مصدر وترجمة)
    pairs = [
        {"i": idx, "source": r.get("source", "").strip(), "translation": r.get("translation", "").strip()}
        for idx, r in enumerate(rows)
        if r.get("source", "").strip() and r.get("translation", "").strip()
    ]

    results = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        results.extend(_review_batch(client, chunk, source_lang, glossary_text, extra_notes, model))

    result_by_i = {r["i"]: r for r in results}

    final = []
    for idx, r in enumerate(rows):
        item = result_by_i.get(idx)
        if item:
            final.append({
                "source": r.get("source", "").strip(),
                "original": r.get("translation", "").strip(),
                "translation": item["translation"],
                "include": bool(r.get("include", True)),
                "file": r.get("file", ""),
                "status": item["status"],
                "reason": item["reason"],
            })
        else:
            final.append({
                "source": r.get("source", "").strip(),
                "original": r.get("translation", "").strip(),
                "translation": r.get("translation", "").strip(),
                "include": bool(r.get("include", True)),
                "file": r.get("file", ""),
                "status": "accepted",
                "reason": "",
            })
    return final


def count_corrected(reviewed_rows: list[dict]) -> int:
    return sum(1 for r in reviewed_rows if r.get("status") == "corrected")
