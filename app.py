# -*- coding: utf-8 -*-
"""
منصة جمع وتدريب الترجمة العربية — واجهة Streamlit

وضعان:
  - رفع للتدريب (المساهمون): رفع الملفات فقط، بدون أي تعديل، وحفظها مباشرة للتدريب.
  - مراجعة وتدقيق (الفريق): جدول تعديل الترجمة قبل اعتمادها.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_core import (
    CHAPTERS_EN,
    CHAPTERS_KO,
    LANG_LABELS,
    OUTPUT_AR,
    TRAINING_DIR,
    archive_submitted_files,
    count_submitted_files,
    load_project,
    load_uploaded,
    list_txt,
    preview_jsonl,
    save_jsonl,
)
from hub_sync import (
    archive_jsonl_in_repo,
    download_jsonl_to_local,
    get_dataset_repo,
    is_cloud,
    upload_jsonl_to_repo,
)
from reviewer import count_corrected, review_pairs


# ---------------------------------------------------------------------------
# التنسيق
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, .stApp, [data-testid="stSidebar"] { direction: rtl; }
        .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { text-align: right; }
        [data-testid="stFileUploader"] { direction: rtl; }
        div[data-baseweb="select"] { text-align: right; }
        .block-container { padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def upload_to_cloud(path) -> None:
    """رفع ملف JSONL محلي إلى المستودع الدائم عند العمل في وضع السحابة."""
    if not is_cloud():
        return
    repo_id = get_dataset_repo()
    if not repo_id:
        st.warning("لن يُرفع الملف للمستودع لأن HF_TOKEN غير مضبوط في Secrets.")
        return
    if upload_jsonl_to_repo(repo_id, path):
        st.success(f"رُفع الملف إلى المستودع الدائم: `{repo_id}`")
    else:
        st.error("فشل رفع الملف إلى المستودع. تحقق من مفتاح HF_TOKEN واسم المستودع في Secrets.")


AUTO_TRAIN_THRESHOLD = 10  # عدد الملفات المطلوب لبدء التدريب التلقائي


def maybe_auto_train(model: str, epochs: int = 3) -> None:
    """بدء التدريب تلقائياً عندما يصل عدد الملفات المرفوعة إلى الحد المطلوب."""
    n = count_submitted_files()
    if n < AUTO_TRAIN_THRESHOLD:
        st.info(f"جُمع {n} ملفاً من أصل {AUTO_TRAIN_THRESHOLD} — سيبدأ التدريب التلقائي عند اكتمال العدد.")
        return

    st.info(f"وصل العدد إلى {AUTO_TRAIN_THRESHOLD} ملفاً — بدء التدريب التلقائي الآن...")
    try:
        from scripts.finetune import start_training
        job_id, n_pairs = start_training(model=model, epochs=epochs)
        st.success(f"بدأت مهمة التدريب: `{job_id}` — عدد الأزواج المدربة: {n_pairs}")
        archived = archive_submitted_files()
        st.caption(f"نُقل {len(archived)} ملفاً إلى مجلد archive/ ليبدأ العد من جديد لدفعة تالية.")
        if is_cloud():
            repo_id = get_dataset_repo()
            if repo_id:
                n_repo = archive_jsonl_in_repo(repo_id, archived)
                st.caption(f"أُرشفت {n_repo} ملفاً في مستودع البيانات `{repo_id}`.")
    except Exception as e:
        st.error(f"فشل بدء التدريب التلقائي: {e}")


# ---------------------------------------------------------------------------
# وضع المساهمين: رفع فقط
# ---------------------------------------------------------------------------
def render_submit_mode() -> None:
    st.header("إرسال ملفات الترجمة للتدريب")
    st.caption("ارفع الفصل المصدري (كوري أو إنجليزي) مع ترجمته العربية. تُحفظ البيانات تلقائياً وتُستخدم لاحقاً في تدريب النموذج.")

    with st.sidebar:
        st.header("الرفع")
        lang = st.radio("لغة النص الأصلي", options=["en", "ko"], format_func=lambda x: LANG_LABELS[x], horizontal=True, key="submit_lang")
        src_uploads = st.file_uploader(
            "ملفات النص الأصلي (.txt)",
            type=["txt"],
            accept_multiple_files=True,
            key="submit_src",
            help="تُطابق الترجمة حسب اسم الملف (مثال: 111.txt يقابل 111.txt).",
        )
        tgt_uploads = st.file_uploader(
            "ملفات الترجمة العربية (.txt)",
            type=["txt"],
            accept_multiple_files=True,
            key="submit_tgt",
        )
        with st.expander("المراجعة التلقائية (اختياري)"):
            enable_review = st.toggle("تفعيل مراجعة النموذج للترجمة", value=False, key="submit_enable_review",
                help="يتطلب رصيد OpenAI. عند تعطيله تُحفظ الترجمات كما أرسلها المساهم دون تصحيح.")
            review_model = st.text_input("نموذج المراجعة", value="gpt-4o-mini-2024-07-18", key="submit_model")
            extra_notes = st.text_area(
                "ملاحظات إضافية للمراجعة (اختياري)",
                value="راجِع الترجمة بحيث تكون عربية فصحى ميسّرة طبيعية، مع الحفاظ على علامات التعجب والاستفهام وروح الحوار.",
                key="submit_notes",
            )

    if st.button("إرسال للتدريب", type="primary", use_container_width=True):
        if not src_uploads and not tgt_uploads:
            st.warning("ارفع ملفاً واحداً على الأقل (النص الأصلي أو الترجمة العربية).")
            st.stop()

        rows = load_uploaded(src_uploads, tgt_uploads)
        if not rows:
            st.error("تعذرت قراءة الملفات. تأكد أن الملفات نصية بصيغة .txt.")
            st.stop()

        n_pairs = sum(1 for r in rows if r.get("source", "").strip() and r.get("translation", "").strip())
        n_missing = sum(1 for r in rows if bool(r.get("source", "").strip()) != bool(r.get("translation", "").strip()))

        if enable_review:
            st.info(f"النموذج يراجع الآن {n_pairs} جملة عبر {review_model}... قد يستغرق بضع دقائق.")
            reviewed = review_pairs(rows, lang, model=review_model, extra_notes=extra_notes)
            n_corrected = count_corrected(reviewed)
        else:
            st.info("حفظ مباشر دون مراجعة تلقائية — تُحفظ الترجمات كما أرسلها المساهم.")
            reviewed = [
                {
                    "source": r.get("source", "").strip(),
                    "original": r.get("translation", "").strip(),
                    "translation": r.get("translation", "").strip(),
                    "include": bool(r.get("include", True)),
                    "file": r.get("file", ""),
                    "status": "accepted",
                    "reason": "",
                }
                for r in rows
            ]
            n_corrected = 0

        fmt = "simple"
        strip_bubbles = True
        instruction = f"ترجم النص التالي من {LANG_LABELS[lang]} إلى العربية."
        file_name = f"submitted_{lang}_{len(list(TRAINING_DIR.glob('*.jsonl'))) + 1}.jsonl"
        path, count = save_jsonl(reviewed, file_name, fmt, instruction, lang, strip_bubbles)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("أزواج سليمة", n_pairs)
        c2.metric("جمل بلا مقابل", n_missing)
        c3.metric("صحّحها النموذج", n_corrected)
        c4.metric("حُفظت للتدريب", count)

        if count:
            if enable_review:
                st.success(f"شكراً لك! راجع النموذج الترجمة وحُفظت {count} جملة في: `{path}`")
            else:
                st.success(f"شكراً لك! حُفظت {count} جملة للتدريب في: `{path}`")
            upload_to_cloud(path)
            maybe_auto_train(model=review_model)
            with st.expander("ما الذي غيّره النموذج؟"):
                if not enable_review:
                    st.caption("المراجعة التلقائية معطّلة — حُفظت الترجمات كما أرسلها المساهم.")
                for r in reviewed:
                    if r.get("status") == "corrected":
                        st.markdown(f"**الأصل:** {r['source']}")
                        st.markdown(f"**قبل:** ~~{r.get('original', '')}~~")
                        st.markdown(f"**بعد:** {r['translation']}")
                        if r.get("reason"):
                            st.caption(f"السبب: {r['reason']}")
                        st.divider()
        else:
            st.warning("لم يُحفظ أي زوج صالح؛ تأكد من إرفاق الترجمة العربية مع النص الأصلي بنفس الاسم.")


# ---------------------------------------------------------------------------
# وضع الفريق: مراجعة وتعديل
# ---------------------------------------------------------------------------
def render_review_mode() -> None:
    with st.sidebar:
        st.header("الملفات")
        source_lang = st.radio("لغة النص الأصلي", options=["en", "ko"], format_func=lambda x: LANG_LABELS[x], horizontal=True, key="review_lang")

        st.subheader("رفع ملفات")
        src_uploads = st.file_uploader(
            "ملفات النص الأصلي (.txt)",
            type=["txt"],
            accept_multiple_files=True,
            key="review_src",
            help="تُطابق ملفات الترجمة حسب اسم الملف (مثال: 111.txt يقابل 111.txt).",
        )
        tgt_uploads = st.file_uploader(
            "ملفات الترجمة العربية (.txt)",
            type=["txt"],
            accept_multiple_files=True,
            key="review_tgt",
        )

        st.subheader("أو من ملفات المشروع")
        src_options = list_txt(CHAPTERS_EN if source_lang == "en" else CHAPTERS_KO)
        tgt_options = list_txt(OUTPUT_AR)
        sel_src = st.multiselect("فصول المصدر", src_options, key="review_sel_src")
        sel_tgt = st.multiselect("الفصول العربية", tgt_options, key="review_sel_tgt")

        if st.button("تحميل ومطابقة الجمل", type="primary", use_container_width=True):
            if src_uploads or tgt_uploads:
                st.session_state.rows = load_uploaded(src_uploads, tgt_uploads)
            elif sel_src or sel_tgt:
                st.session_state.rows = load_project(sel_src, sel_tgt, source_lang)
            else:
                st.session_state.rows = []
            if st.session_state.rows:
                st.success(f"تم تحميل {len(st.session_state.rows)} جملة.")
            else:
                st.info("لم يُعثر على جمل. ارفع ملفات أو اختر من المشروع.")

    rows = st.session_state.get("rows", [])
    if not rows:
        st.info("اختر ملفات الفصل من الشريط الجانبي ثم اضغط «تحميل ومطابقة الجمل».")
        return

    df = pd.DataFrame(rows)
    n_src = int((df["source"].str.strip() != "").sum())
    n_tgt = int((df["translation"].str.strip() != "").sum())
    n_inc = int(df["include"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("جمل المصدر", n_src)
    c2.metric("جمل الترجمة", n_tgt)
    c3.metric("مشمولة في التدريب", n_inc)
    c4.metric("الإجمالي", len(df))

    st.subheader("مراجعة الجمل")
    st.caption("عدّل عمود «الترجمة العربية» مباشرة، وألغِ تضمين أي جملة من عمود «تضمين».")

    edited = st.data_editor(
        df,
        key="editor",
        height=520,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        column_config={
            "file": st.column_config.TextColumn("الملف", disabled=True, width="small"),
            "source": st.column_config.TextColumn("النص الأصلي", disabled=True, width="large"),
            "translation": st.column_config.TextColumn("الترجمة العربية", width="large"),
            "include": st.column_config.CheckboxColumn("تضمين", width="small"),
        },
    )
    st.session_state.rows = edited.to_dict("records")

    st.divider()
    st.subheader("الحفظ والتدريب")

    col1, col2, col3 = st.columns([2, 2, 2])
    fmt = col1.selectbox("تنسيق JSONL", options=["alpaca", "simple"], format_func=lambda x: "ألباكا (instruction/input/output)" if x == "alpaca" else "بسيط (source/translation)", key="review_fmt")
    strip_bubbles = col2.toggle("إزالة رموز الفقاعات ([]: (): :: …)", value=True, key="review_strip")
    instruction = col3.text_input("تعليمات التدريب (لتنسيق ألباكا)", value=f"ترجم النص التالي من {LANG_LABELS[source_lang]} إلى العربية.", key="review_instruction")

    with st.expander("معاينة البيانات قبل الحفظ"):
        preview_lines = preview_jsonl(st.session_state.rows, fmt, instruction, source_lang, strip_bubbles)
        if preview_lines:
            st.code("\n".join(preview_lines), language="json")
        else:
            st.caption("لا توجد جمل مشمولة بعد.")

    col_a, col_b = st.columns([2, 3])
    file_name = col_a.text_input("اسم ملف JSONL", value="training_data.jsonl", key="review_filename")

    if st.button("اعتماد للحفظ والتدريب", type="primary"):
        path, count = save_jsonl(st.session_state.rows, file_name, fmt, instruction, source_lang, strip_bubbles)
        if count:
            st.success(f"تم الحفظ بنجاح: {path} — {count} زوجاً جاهزاً للتدريب.")
            upload_to_cloud(path)
            st.download_button("تحميل الملف", data=path.read_bytes(), file_name=file_name, mime="application/jsonl")
        else:
            st.warning("لا توجد جمل مشمولة؛ فعّل تضمين بعض الجمل أولاً.")


# ---------------------------------------------------------------------------
# نقطة الدخول
# ---------------------------------------------------------------------------
st.set_page_config(page_title="منصة جمع وتدريب الترجمة", page_icon="📖", layout="wide")
inject_css()

st.title("منصة جمع وتدريب الترجمة العربية")
st.caption("يُرسل المساهمون ملفات الترجمة للتدريب، ويتولى الفريق مراجعتها واعتمادها.")

# الوضع السحابي: مزامنة بيانات التدريب من مستودع Hugging Face
if is_cloud():
    repo_id = get_dataset_repo()
    if repo_id:
        n = download_jsonl_to_local(repo_id)
        st.caption(f"وضع السحابة: تمت مزامنة {n} ملف JSONL من المستودع `{repo_id}`.")
    else:
        st.warning("وضع السحابة نشط لكن HF_TOKEN غير مضبوط. أضف المفتاح في Secrets حتى تُرفع البيانات للمستودع الدائم.")

mode = st.radio(
    "الوضع",
    options=["رفع للتدريب (المساهمون)", "مراجعة وتدقيق (الفريق)"],
    horizontal=True,
    key="app_mode",
)

if mode.startswith("رفع"):
    render_submit_mode()
else:
    render_review_mode()
