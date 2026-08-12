#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""مزامنة بيانات التدريب مع مستودع بيانات Hugging Face.

عند نشر الواجهة على Hugging Face Spaces، المجلد المحلي training_data/ مؤقت
ويُمحى عند كل إعادة تشغيل. لذا تُرفع الملفات إلى مستودع بيانات (Dataset) دائم
عبر API، وتُعاد مزامنتها عند تشغيل الواجهة.

الإعدادات (تُضبط كـ Secrets في إعدادات الـ Space):
  HF_TOKEN          مفتاح الوصول من Hugging Face (مطلوب)
  HF_DATASET_REPO   معرّف المستودع، مثل: username/manhwa-ar-training
                    (اختياري؛ الافتراضي username/manhwa-ar-training)
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "training_data"
DEFAULT_SUFFIX = "manhwa-ar-training"


def is_cloud() -> bool:
    """هل نعمل في بيئة سحابية (Hugging Face Spaces أو Streamlit Cloud)؟

    يُكتشف عبر وجود SPACE_ID (في Hugging Face Spaces) أو HF_TOKEN
    (يُضبط كسرّ في أي منصة استضافة، ولا يوجد محلياً في العادة).
    """
    if os.environ.get("SPACE_ID", "").strip():
        return True
    return bool(os.environ.get("HF_TOKEN", "").strip())


def get_hf_token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()


def get_dataset_repo() -> str:
    """تحديد معرّف مستودع البيانات. يقرأه من HF_DATASET_REPO أو يولّده من اسم المستخدم."""
    repo = os.environ.get("HF_DATASET_REPO", "").strip()
    if repo:
        return repo
    token = get_hf_token()
    if not token:
        return ""
    try:
        from huggingface_hub import HfApi
        user = HfApi(token=token).whoami()["name"]
        return f"{user}/{DEFAULT_SUFFIX}"
    except Exception:
        return ""


def get_api():
    from huggingface_hub import HfApi
    return HfApi(token=get_hf_token() or None)


def download_jsonl_to_local(repo_id: str) -> int:
    """تنزيل كل ملفات JSONL من المستودع إلى training_data/ المحلي. يعيد عدد الملفات."""
    from huggingface_hub import hf_hub_download

    if not repo_id:
        return 0
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    api = get_api()
    count = 0
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return 0
    for name in files:
        if not name.endswith(".jsonl") or name.startswith("archive/"):
            continue  # الأرشفة لا تُعاد للجذر
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=name,
                local_dir=str(TRAINING_DIR),
            )
            count += 1
        except Exception:
            continue
    return count


def archive_jsonl_in_repo(repo_id: str, file_names: list[str]) -> int:
    """نقل ملفات JSONL من جذر المستودع إلى مجلد archive/ بعد انتهاء التدريب.

    يُستخدم لتفادي إعادة تدريب نفس البيانات عند المزامنة التالية.
    يعيد عدد الملفات المنقولة بنجاح.
    """
    if not repo_id or not file_names:
        return 0
    api = get_api()
    moved = 0
    for name in file_names:
        if not name.endswith(".jsonl") or name.startswith("archive/"):
            continue
        local = TRAINING_DIR / "archive" / name
        try:
            # أولاً نرفع نسخة داخل مجلد archive/ ثم نحذف الأصل من الجذر
            if local.exists():
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=f"archive/{name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"أرشفة {name} بعد التدريب",
                )
            api.delete_file(
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"حذف {name} من الجذر بعد التدريب",
            )
            moved += 1
        except Exception:
            continue
    return moved


def upload_jsonl_to_repo(repo_id: str, local_path: Path, commit_message: str | None = None) -> bool:
    """رفع ملف JSONL محلي إلى مستودع البيانات. يعيد True عند النجاح."""
    if not repo_id or not local_path.exists():
        return False
    api = get_api()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception:
        return False
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message or f"إضافة {local_path.name}",
        )
        return True
    except Exception:
        return False


def repo_stats(repo_id: str) -> dict:
    """إحصائيات بيانات التدريب في المستودع (بالقراءة فقط).

    يعيد قاموساً:
      - files: عدد ملفات JSONL المجمّعة (غير المؤرشفة)
      - archived: عدد الملفات المؤرشفة (المدربة سابقاً)
      - pairs: إجمالي الأزواج (source/translation) عبر الملفات المجمّعة
      - trained: إجمالي الأزواج في الملفات المؤرشفة
      - pairs_by_lang: عدّاد الأزواج حسب لغة المصدر
    """
    if not repo_id:
        return {"files": 0, "archived": 0, "pairs": 0, "trained": 0, "pairs_by_lang": {}}
    api = get_api()
    stats = {"files": 0, "archived": 0, "pairs": 0, "trained": 0, "pairs_by_lang": {}}
    try:
        names = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return stats
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        is_archive = name.startswith("archive/")
        pairs, langs = _count_jsonl_pairs(api, repo_id, name)
        if is_archive:
            stats["archived"] += 1
            stats["trained"] += pairs
        else:
            stats["files"] += 1
            stats["pairs"] += pairs
            for lang, c in langs.items():
                stats["pairs_by_lang"][lang] = stats["pairs_by_lang"].get(lang, 0) + c
    return stats


def _count_jsonl_pairs(api, repo_id: str, name: str) -> tuple[int, dict]:
    """قراءة ملف JSONL من المستودع وإحصاء الأزواج وتوزيعها حسب لغة المصدر."""
    pairs = 0
    langs: dict[str, int] = {}
    try:
        content = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=name)
        with open(content, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if (rec.get("source") or "").strip() and (rec.get("translation") or "").strip():
                    pairs += 1
                    lang = rec.get("source_language") or "?"
                    langs[lang] = langs.get(lang, 0) + 1
    except Exception:
        pass
    return pairs, langs
