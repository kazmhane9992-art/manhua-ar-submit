#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""تدريب دقيق مجاني لنموذج ترجمة (Qwen) على بياناتك عبر Google Colab.

كيف تعمل:
  1) افتح https://colab.research.google.com واختر GPU (Runtime → Change runtime type → T4).
  2) الصق محتوى هذا الملف في خلية واحدة (أو ارفعه ثم شغّل: !python colab_finetune.py).
  3) عدّل المتغيرات الثلاثة أدناه: HF_TOKEN وDATASET_REPO واسم النموذج إن رغبت.
  4) شغّل الخلية وانتظر — المدة التقريبية للتدريب: 20-60 دقيقة (حسب حجم البيانات).
  5) يُرفع النموذج المدرب تلقائياً إلى مستودعك على Hugging Face.

المتطلبات: حساب Hugging Face مجاني + بيانات مرفوعة في مستودعك (كما في التطبيق).
"""

import json
import os
from pathlib import Path

# ═══════════════════════ إعداداتك هنا ═══════════════════════
HF_TOKEN = ""            # ضع توكن Hugging Face هنا (مثل hf_...)
DATASET_REPO = ""        # مستودع بياناتك، مثال: "kzome/manhua-ar-training"
MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"   # النموذج الأساسي
OUTPUT_REPO = ""         # مستودع الناتج، مثال: "kzome/manhua-ar-model" (يُنشأ تلقائياً)
MAX_SEQ_LEN = 1024
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
WARMUP_STEPS = 10
# ════════════════════════════════════════════════════════════

assert HF_TOKEN, "ضع توكن Hugging Face في HF_TOKEN."
assert DATASET_REPO, "ضع اسم مستودع بياناتك في DATASET_REPO."
assert OUTPUT_REPO, "ضع اسم مستودع الناتج في OUTPUT_REPO."

print("== 1) تثبيت المكتبات (Unsloth) ==")
os.system("pip install --upgrade --quiet unsloth[colab-new] && pip install --upgrade --quiet bitsandbytes xformers trl peft accelerate datasets")

print("== 2) تنزيل بيانات التدريب من المستودع ==")
from huggingface_hub import hf_hub_download
DATA_DIR = Path("/content/data")
DATA_DIR.mkdir(exist_ok=True)

api_url = f"https://huggingface.co/api/datasets/{DATASET_REPO}"
import urllib.request
with urllib.request.urlopen(api_url) as r:
    info = json.load(r)

downloaded = 0
for f in info.get("siblings", []):
    name = f.get("rfilename", "")
    if not name.endswith(".jsonl") or name.startswith("archive/"):
        continue
    hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename=name,
        local_dir=str(DATA_DIR), token=HF_TOKEN,
    )
    downloaded += 1
print(f"تم تنزيل {downloaded} ملف JSONL.")

jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
assert jsonl_files, "لا توجد ملفات JSONL في المستودع!"

print("== 3) تجهيز البيانات بتنسيق المحادثة ==")
SYSTEM_PROMPT = (
    "أنت مترجم محترف متخصص في المانجا والمانهوا، تنقل المعنى والروح "
    "إلى عربية فصحى ميسّرة. لا تترجم حرفياً؛ أعد الصياغة بأسلوب طبيعي "
    "مع الحفاظ على علامات التعجب والاستفهام وروح الحوار."
)

pairs = []
for path in jsonl_files:
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        src = (rec.get("source") or "").strip()
        tgt = (rec.get("translation") or "").strip()
        if src and tgt:
            pairs.append((src, tgt))
print(f"إجمالي الأزواج: {len(pairs)}")

from datasets import Dataset

def format_example(example):
    return {
        "text": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\nترجم إلى العربية: {example['source']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['translation']}<|im_end|>\n",
    }

dataset = Dataset.from_list([{"source": s, "translation": t} for s, t in pairs])
dataset = dataset.map(format_example)

print("== 4) تحميل النموذج (Unsloth) ==")
from unsloth import FastLanguageModel, is_bfloat16_supported
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

print("== 5) تجهيز LoRA ==")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print("== 6) التدريب ==")
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer.train()

print("== 7) حفظ النموذج ==")
model.save_pretrained_merged("manhua-ar-model", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged(OUTPUT_REPO, tokenizer, save_method="merged_16bit", token=HF_TOKEN)

print("== 8) اختبار سريع ==")
FastLanguageModel.for_inference(model)
prompt = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\nترجم إلى العربية: "
test_src = pairs[0][0] if pairs else "Hello"
inputs = tokenizer(prompt + test_src + "<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.95)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

print("\nتم بنجاح! النموذج المدرب في مستودعك:")
print(f"  https://huggingface.co/{OUTPUT_REPO}")
print("نموذج صغير خفيف للاستخدام اليومي اختياري: عدّل save_method إلى 'merged_16bit' أو استخدم 'gguf' لتحويله لـ GGUF.")
