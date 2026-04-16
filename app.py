import streamlit as st
from transformers import pipeline
from gtts import gTTS
import json
import uuid
import os

st.set_page_config(page_title="Medical AI Assistant", layout="wide")

st.title("🩺 Medical AI Assistant")
st.markdown("⚠️ هذا النظام يقدم معلومات عامة وليس تشخيصًا طبيًا")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

model = load_model()

# ---------------- HISTORY ----------------
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

history = load_history()

# ---------------- UI ----------------
symptoms = st.text_area("🧾 اكتب الأعراض:", placeholder="مثال: صداع، حرارة، كحة")

col1, col2 = st.columns(2)

# ---------------- FUNCTION ----------------
def analyze(symptoms):
    prompt = f"""
أنت مساعد طبي ذكي.

المطلوب:
1- تحديد نوع الحالة (تنفسية / هضمية / عصبية / جلدية / أخرى)
2- ذكر أسباب محتملة
3- تحديد مستوى الخطورة (خفيف / متوسط / طارئ)
4- نصيحة عامة

الأعراض: {symptoms}

اكتب الإجابة باللغة العربية الفصحى بشكل منظم.
"""

    result = model(prompt, max_new_tokens=200)[0]["generated_text"]
    return result

# ---------------- RUN ----------------
if st.button("🔍 تحليل الأعراض"):

    if symptoms.strip() == "":
        st.warning("من فضلك اكتب الأعراض")
    else:
        result = analyze(symptoms)

        st.subheader("📋 النتيجة")
        st.write(result)

        # 🔊 صوت
        tts = gTTS(text=result, lang="ar")
        audio_file = f"voice_{uuid.uuid4().hex}.mp3"
        tts.save(audio_file)

        st.audio(audio_file)

        # 🗂️ حفظ التاريخ
        history.append({
            "symptoms": symptoms,
            "result": result
        })
        save_history(history)

# ---------------- HISTORY UI ----------------
st.sidebar.title("📜 تاريخ الحالات")

for item in reversed(history[-10:]):
    st.sidebar.markdown("-----")
    st.sidebar.write("🧾 الأعراض:")
    st.sidebar.write(item["symptoms"])
    st.sidebar.write("📋 النتيجة:")
    st.sidebar.write(item["result"])
