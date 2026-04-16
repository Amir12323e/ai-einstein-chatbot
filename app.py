import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import uuid

st.title("🩺 Medical AI Assistant")

st.warning("⚠️ هذا النظام معلومات عامة وليس تشخيص طبي")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-400M-distill"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

symptoms = st.text_area("🧾 اكتب الأعراض:")

# ---------------- ANALYSIS ----------------
def analyze(symptoms):
    prompt = f"""
أنت مساعد طبي ذكي.
حلل الأعراض التالية واذكر احتمالات عامة فقط.
لا تقدم تشخيص نهائي.

الأعراض: {symptoms}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- RUN ----------------
if st.button("تحليل"):
    if not symptoms.strip():
        st.warning("اكتب الأعراض أولاً")
    else:
        result = analyze(symptoms)

        st.subheader("📋 النتيجة")
        st.write(result)

        # 🔊 صوت
        tts = gTTS(text=result, lang="ar")
        audio_file = f"voice_{uuid.uuid4().hex}.mp3"
        tts.save(audio_file)

        st.audio(audio_file)
