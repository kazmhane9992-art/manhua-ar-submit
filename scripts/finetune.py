#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""سكريبت التدريب الدقيق (Fine-tuning) للنموذج على بيانات الترجمة.

يقرأ ملفات JSONL من مجلد training_data (التي حُفظت من واجهة المستخدم)،
يدمجها في ملف واحد بتنسيق رسائل OpenAI (System / User / Assistant)،
ثم يرفع الملف ويبدأ مهمة تدريب عبر API، مع أمر لمتابعة حالة التدريب.

الاستخدام:
  1) ضع مفتاح API في متغير البيئة:
       set OPENAI_API_KEY=sk-...
     أو أنشئ ملف .env في مجلد المشروع يحتوي:
       OPENAI_API_KEY=sk-...

  2) تجهيز ملف التدريب المدمج فقط:
       python scripts/finetune.py prepare

  3) تجهيز + رفع + بدء التدريب:
       python scripts/finetune.py train --model gpt-4o-mini-2024-07-18

  4) متابعة حالة مهمة تدريب محددة:
       python scripts/finetune.py status --job-id ftjob_XXXXXXXX

  5) متابعة حالة آخر مهمة بدأت من هذا السكريبت:
       python scripts/finetune.py status

  6) قائمة المهام الأخيرة:
       python scripts/finetune.py list
"""

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# المسارات الثابتة
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training_data"
MERGED_FILE = TRAINING_DIR / "merged_finetune.jsonl"
LAST_JOB_FILE = TRAINING_DIR / ".last_job_id"

# نص رسالة النظام (System) — عدّله كما تشاء لتوجيه سلوك النموذج المدرَّب.
SYSTEM_PROMPT = (
    "أنت مترجم محترف متخصص في ترجمة المانجا والمانهوا من "
    "الإنجليزية والكورية إلى العربية الفصحى الميسّرة. "
    "ترجم الجملة التالية إلى العربية بأسلوب طبيعي وسلس، مع الحفاظ على "
    "الإيقاع والمشاعر، وعدم الترجمة الحرفية."
)


# ---------------------------------------------------------------------------
# 1) القراءة والدمج والتنسيق
# ---------------------------------------------------------------------------
def read_jsonl_lines(path: Path) -> list[dict]:
    """قراءة سطور ملف JSONL وإرجاعها كقائمة قواميس."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [تحذير] سطر غير صالح في {path.name}، تم تجاوزه.")
    return rows


def extract_pair(record: dict) -> dict | None:
    """استخراج زوج (مصدر، ترجمة) من أي تنسيق حفظته الواجهة.

    يدعم تنسيقين:
      - "simple":  {"source":..., "translation":...}
      - "alpaca":  {"input":..., "output":...}
    """
    src = record.get("source") or record.get("input")
    tgt = record.get("translation") or record.get("output")
    if not src or not tgt:
        return None
    src = str(src).strip()
    tgt = str(tgt).strip()
    if not src or not tgt:
        return None
    return {"source": src, "translation": tgt}


def to_chat_message(pair: dict, system_prompt: str = SYSTEM_PROMPT) -> dict:
    """تحويل الزوج إلى رسائل OpenAI القياسية.

    التنسيق:
      {"messages": [
          {"role": "system",    "content": <تعليمات النظام>},
          {"role": "user",      "content": <النص الأصلي en/ko>},
          {"role": "assistant", "content": <الترجمة العربية>}
      ]}
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["source"]},
            {"role": "assistant", "content": pair["translation"]},
        ]
    }


def merge_all_files() -> tuple[list[dict], int]:
    """دمج كل ملفات JSONL في training_data (عدا الملف المدمج نفسه) وتنسيقها كرسائل.

    يعيد (قائمة الرسائل، عدد الأزواج الفريدة).
    """
    if not TRAINING_DIR.exists():
        raise FileNotFoundError(f"المجلد {TRAINING_DIR} غير موجود. احفظ بيانات من الواجهة أولاً.")

    jsonl_files = sorted(
        p for p in TRAINING_DIR.glob("*.jsonl")
        if p.name != MERGED_FILE.name
    )
    if not jsonl_files:
        raise FileNotFoundError(f"لا توجد ملفات JSONL في {TRAINING_DIR}.")

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in jsonl_files:
        records = read_jsonl_lines(path)
        added = 0
        for rec in records:
            pair = extract_pair(rec)
            if pair is None:
                continue
            key = (pair["source"], pair["translation"])
            if key in seen:
                continue  # إزالة التكرارات
            seen.add(key)
            pairs.append(pair)
            added += 1
        print(f"  {path.name}: {len(records)} سطراً -> {added} زوجاً صالحاً.")

    messages = [to_chat_message(p) for p in pairs]
    with MERGED_FILE.open("w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\nالمجموع: {len(pairs)} زوجاً بعد إزالة التكرارات.")
    print(f"حُفظ الملف المدمج في: {MERGED_FILE}")
    return messages, len(pairs)


def prepare(args) -> None:
    """دمج كل ملفات JSONL في training_data وتحويلها لتنسيق الرسائل."""
    merge_all_files()


# ---------------------------------------------------------------------------
# 2) الاتصال بـ OpenAI API
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    """قراءة مفتاح API من متغير البيئة OPENAI_API_KEY."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print(
            "خطأ: مفتاح OPENAI_API_KEY غير مضبوط.\n"
            "اضبطه بأحد الطريقتين:\n"
            "  1) في الطرفية:  set OPENAI_API_KEY=sk-...  (على ويندوز)\n"
            "  2) أنشئ ملف .env في مجلد المشروع بالمحتوى:  OPENAI_API_KEY=sk-...\n"
            "ثم أعد تشغيل السكريبت."
        )
        sys.exit(1)
    return key


def get_client():
    """إنشاء عميل OpenAI. يتطلب تثبيت:  pip install openai"""
    try:
        from openai import OpenAI
    except ImportError:
        print("مكتبة openai غير مثبتة. ثبّتها بالأمر:\n  pip install openai")
        sys.exit(1)
    return OpenAI(api_key=get_api_key())


def start_training(model: str = "gpt-4o-mini-2024-07-18", epochs: int = 3, suffix: str = "manhwa-ar", merge: bool = True) -> tuple[str, int]:
    """تجهيز البيانات ورفعها وبدء مهمة تدريب دقيق.

    - merge=True: يدمج ملفات training_data أولاً ثم يرفع.
    - يعيد (معرّف المهمة، عدد الأزواج المدربة).
    """
    client = get_client()
    if merge:
        _, n_pairs = merge_all_files()
    else:
        n_pairs = 0

    if not MERGED_FILE.exists():
        raise FileNotFoundError("الملف المدمج غير موجود. أعد المحاولة مع merge=True.")

    print(f"رفع الملف: {MERGED_FILE} ...")
    with MERGED_FILE.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    print(f"معرّف الملف: {uploaded.id}")

    print(f"بدء مهمة التدريب بالنموذج: {model} ...")
    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=model,
        suffix=suffix or None,
        hyperparameters={"n_epochs": epochs},
    )
    print(f"تم إنشاء المهمة: {job.id}")
    with LAST_JOB_FILE.open("w", encoding="utf-8") as f:
        f.write(job.id)
    print(f"متابعة الحالة بالأمر:\n  python scripts/finetune.py status")
    return job.id, n_pairs


def start_finetune(client, args) -> str:
    """واجهة سطر الأوامر لبدء التدريب (تستخدم start_training)."""
    job_id, n_pairs = start_training(
        model=args.model,
        epochs=args.epochs,
        suffix=args.suffix,
        merge=not args.skip_prepare,
    )
    return job_id


def show_job_status(client, job_id: str) -> None:
    """استرجاع وطباعة حالة مهمة التدريب (جارية / نجحت / فشلت)."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    print("=" * 50)
    print(f"معرّف المهمة:    {job.id}")
    print(f"الحالة:          {job.status}")
    print(f"النموذج:         {job.model}")
    print(f"نموذج المخرجات:  {getattr(job, 'fine_tuned_model', None) or '—'}")
    print(f"ملف التدريب:     {job.training_file}")
    if job.error:
        print(f"الخطأ:           {job.error}")
    print("=" * 50)

    if job.status == "succeeded":
        print("\nتم التدريب بنجاح! النموذج المدرَّب:")
        print(f"  {job.fine_tuned_model}")
        print("استخدامه:\n  model = client.chat.completions.create(model='<النموذج>', messages=[...])")
    elif job.status == "failed":
        print("\nفشلت المهمة. راجع حقل الخطأ بالأعلى.")
    elif job.status in {"queued", "running"}:
        print("\nالمهمة ما زالت قيد المعالجة... تحقق لاحقاً بنفس الأمر.")
    else:
        print(f"\nحالة غير معروفة: {job.status}")


def status(args) -> None:
    """متابعة حالة آخر مهمة، أو مهمة محددة عبر --job-id."""
    client = get_client()
    job_id = args.job_id
    if not job_id:
        if LAST_JOB_FILE.exists():
            job_id = LAST_JOB_FILE.read_text(encoding="utf-8").strip()
        else:
            print("لا توجد مهمة سابقة. مرّر معرّف المهمة: python scripts/finetune.py status --job-id ftjob_...")
            sys.exit(1)
    show_job_status(client, job_id)


def list_jobs(args) -> None:
    """عرض آخر المهام المسجلة."""
    client = get_client()
    jobs = client.fine_tuning.jobs.list(limit=args.limit)
    for job in jobs.data:
        print(f"{job.id}  |  {job.status:<10}  |  {job.model}")


# ---------------------------------------------------------------------------
# 3) نقطة الدخول
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="التدريب الدقيق على بيانات الترجمة")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="دمج ملفات JSONL وتحويلها لتنسيق الرسائل فقط")
    p_prepare.set_defaults(func=prepare)

    p_train = sub.add_parser("train", help="تجهيز + رفع + بدء مهمة التدريب")
    p_train.add_argument("--model", default="gpt-4o-mini-2024-07-18", help="النموذج الأساسي للتدريب")
    p_train.add_argument("--epochs", type=int, default=3, help="عدد مرات التدريب (افتراضي 3)")
    p_train.add_argument("--suffix", default="manhwa-ar", help="لاحقة اسم النموذج الناتج")
    p_train.add_argument("--skip-prepare", action="store_true", help="لا تعدّ الملف، استخدم الملف المدمج الموجود")
    p_train.set_defaults(func=start_finetune)

    p_status = sub.add_parser("status", help="متابعة حالة التدريب")
    p_status.add_argument("--job-id", default=None, help="معرّف المهمة (اختياري؛ الافتراضي آخر مهمة)")
    p_status.set_defaults(func=status)

    p_list = sub.add_parser("list", help="عرض آخر المهام")
    p_list.add_argument("--limit", type=int, default=5, help="عدد المهام (افتراضي 5)")
    p_list.set_defaults(func=list_jobs)

    args = parser.parse_args()

    if args.command == "train":
        client = get_client()
        if not args.skip_prepare:
            prepare(args)
        start_finetune(client, args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
