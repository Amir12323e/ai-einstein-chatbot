import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import uuid

st.title("🧠 محادثة الشخصيات التاريخية")

@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-400M-distill"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

persona = st.selectbox(
    "اختر الشخصية:",
    ["أينشتاين", "نيوتن", "ابن سينا", "صلاح الدين الأيوبي"]
)

question = st.text_input("اكتب سؤالك:")

def ask(persona, question):
    if not question.strip():
        return "اكتب سؤال من فضلك"

    prompt = f"""
أنت شخصية تاريخية اسمها {persona}.
أجب دائماً باللغة العربية الفصحى فقط.
اجعل الإجابة واضحة وبسيطة وصحيحة قدر الإمكان.

السؤال: {question}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

if st.button("اسأل"):
    answer = ask(persona, question)

    st.success("🧠 الإجابة:")
    st.write(answer)

    # 🔊 تحويل النص لصوت
    tts = gTTS(text=answer, lang='ar')

    audio_file = f"voice_{uuid.uuid4().hex}.mp3"
    tts.save(audio_file)

    st.audio(audio_file)
