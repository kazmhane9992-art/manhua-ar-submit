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

- **تلقائياً**: عندما يصل عدد ملفات الرفع إلى **10** في مجلد `training_data/`، تبدأ الواجهة التدريب تلقائياً ثم تنقل الملفات إلى `training_data/archive/` لبدء دورة جديدة. (غيّر `AUTO_TRAIN_THRESHOLD` في `app.py` لتعديل الحد.)
- **يدوياً من الطرفية**:

```bash
python scripts/finetune.py prepare   # دمج الملفات في merged_finetune.jsonl
python scripts/finetune.py train     # رفع الملف وبدء مهمة التدريب
python scripts/finetune.py status    # متابعة حالة المهمة
```

ملاحظة: يعمل التدريب التلقائي فقط عند توفر `OPENAI_API_KEY` في Secrets.
