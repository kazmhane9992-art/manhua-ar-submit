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

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "training_data"
DEFAULT_SUFFIX = "manhwa-ar-training"


def is_cloud() -> bool:
    """هل نعمل داخل Hugging Face Spaces؟"""
    return bool(os.environ.get("SPACE_ID", "").strip())


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
        if not name.endswith(".jsonl"):
            continue
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


def upload_jsonl_to_repo(repo_id: str, local_path: Path, commit_message: str | None = None) -> bool:
    """رفع ملف JSONL محلي إلى مستودع البيانات. يعيد True عند النجاح."""
    from huggingface_hub import RepositoryNotFoundError, HfApi

    if not repo_id or not local_path.exists():
        return False
    api = get_api()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
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
