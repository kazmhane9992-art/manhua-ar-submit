---
title: منصة مراجعة الترجمة العربية
emoji: 📖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.61.1
app_file: app.py
pinned: false
license: mit
---

# منصة مراجعة الترجمة العربية

واجهة لرفع فصول المانجا، مراجعة الترجمة جملةً جملة، وتجهيز بيانات JSONL جاهزة للتدريب الدقيق.

## الإعداد (Secrets)

من تبويب **Settings → Variables and secrets** أضف:

| المتغير | الوصف | مطلوب |
|---|---|---|
| `HF_TOKEN` | مفتاح وصول Hugging Face (من https://huggingface.co/settings/tokens، بصلاحية كتابة) | نعم |
| `HF_DATASET_REPO` | مستودع البيانات، مثل `username/manhwa-ar-training` | اختياري — يُنشأ تلقائياً باسم `username/manhwa-ar-training` |
| `OPENAI_API_KEY` | مفتاح OpenAI للمراجعة التلقائية للترجمة (من https://platform.openai.com/api-keys) | نعم |

تُرفع ملفات JSONL المعتمدة تلقائياً إلى مستودع البيانات، وتُعاد مزامنتها عند كل تشغيل. وعند إرسال المساهم ملفات، يراجعها نموذج OpenAI تلقائياً (يصحّح الصياغة مع الحفاظ على المعنى والقاموس) ثم تُحفظ النتائج.

## الاستخدام المحلي

```bash
pip install -r requirements.txt
streamlit run app.py
```

## التدريب

### مجاناً (دون رصيد OpenAI)

[![افتح في Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kazmhane9992-art/manhua-ar-submit/blob/main/training/colab_finetune.ipynb)

1. اضغط الزر أعلاه — يفتح دفتر `training/colab_finetune.ipynb` مباشرة في Colab.
2. في Colab: **Runtime → Change runtime type → T4 GPU**.
3. في الخلية الأولى عدّل الثوابت: `HF_TOKEN` و`DATASET_REPO` و`OUTPUT_REPO`.
4. شغّل الخلايا بالترتيب وانتظر (20–60 دقيقة) — يُدرَّب نموذج Qwen على بيانات مستودعك ويُرفع ناتجه لمستودعك تلقائياً.

السكربت النصي أيضاً متاح: `training/colab_finetune.py`.

### تلقائياً (يتطلب رصيد OpenAI)

- عندما يصل عدد ملفات الرفع إلى **10** في مجلد `training_data/`، تبدأ الواجهة التدريب تلقائياً ثم تنقل الملفات إلى `training_data/archive/` لبدء دورة جديدة. (غيّر `AUTO_TRAIN_THRESHOLD` في `app.py` لتعديل الحد.)
- يدوياً من الطرفية:

```bash
python scripts/finetune.py prepare   # دمج الملفات في merged_finetune.jsonl
python scripts/finetune.py train     # رفع الملف وبدء مهمة التدريب
python scripts/finetune.py status    # متابعة حالة المهمة
```

ملاحظة: يعمل التدريب التلقائي ومراجعة الترجمة فقط عند توفر رصيد في `OPENAI_API_KEY`.
