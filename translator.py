# -*- coding: utf-8 -*-
"""الترجمة عبر النموذج المدرب مفتوح المصدر (المستضاف على Hugging Face).

يستخدم نموذج الترجمة المدرَّب (مثل kzome/manhua-ar-model) عبر
Hugging Face Inference API — مجاني ضمن حدود الاستخدام، ولا يتطلب
أي رصيد OpenAI.

الإعدادات:
  HF_TRANSLATION_MODEL   معرف النموذج، مثال: kzome/manhua-ar-model
  HF_TOKEN               مفتاح Hugging Face (قراءة كافية؛ يُستخدم أيضاً)
"""

import os

SYSTEM_PROMPT = (
    "أنت مترجم محترف متخصص في المانجا والمانهوا، تنقل المعنى والروح "
    "إلى عربية فصحى ميسّرة. لا تترجم حرفياً؛ أعد الصياغة بأسلوب طبيعي "
    "مع الحفاظ على علامات التعجب والاستفهام وروح الحوار."
)

DEFAULT_MODEL = "kzome/manhua-ar-model"


def get_translation_model() -> str:
    return os.environ.get("HF_TRANSLATION_MODEL", "").strip() or DEFAULT_MODEL


def translate_text(text: str, source_lang: str = "auto", model: str | None = None) -> str:
    """ترجمة نص إلى العربية عبر النموذج المدرب. يعيد النص المترجم أو نصاً فارغاً عند الفشل."""
    text = (text or "").strip()
    if not text:
        return ""
    model_id = model or get_translation_model()
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return ""
    token = os.environ.get("HF_TOKEN", "").strip() or None
    try:
        client = InferenceClient(model=model_id, token=token)
        lang_hint = {"en": "الإنجليزية", "ko": "الكورية", "auto": ""}.get(source_lang, "")
        user_msg = f"ترجم إلى العربية: {text}" + (f" (الأصل بالكورية)" if lang_hint == "الكورية" else "")
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        result = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        out = (result or "").strip()
        # إنهاء عند أول <|im_end|> إن ظهر
        if "<|im_end|>" in out:
            out = out.split("<|im_end|>")[0].strip()
        return out
    except Exception as e:
        print(f"[translator] فشل الاستدعاء ({model_id}): {e}")
        return ""
