import streamlit as st
from transformers import pipeline
from gtts import gTTS
import uuid

st.title("🧠 أينشتاين الذكي")

question = st.text_input("اسأل أينشتاين:")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="aubmindlab/aragpt2-base")

generator = load_model()

def chat(question):
    if not question.strip():
        return "من فضلك اكتب سؤال"

    prompt = f"أجب بشكل ذكي وبسيط:\nالسؤال: {question}\nالإجابة:"

    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return result[0]["generated_text"]

if st.button("Ask"):
    answer = chat(question)
    st.write(answer)

    filename = f"voice_{uuid.uuid4().hex}.mp3"
    tts = gTTS(answer, lang='ar')
    tts.save(filename)

    st.audio(filename)
