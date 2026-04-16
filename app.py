import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

st.title("🧠 أينشتاين الذكي")

question = st.text_input("اسأل أينشتاين:")

generator = pipeline("text-generation", model="gpt2")

def chat(question):
    if question.strip() == "":
        return "من فضلك اكتب سؤال"

    prompt = f"""
    أنت العالم ألبرت أينشتاين.
    أجب على السؤال التالي باللغة العربية بأسلوب بسيط وذكي:

    السؤال: {question}
    """

    result = generator(
        prompt,
        max_length=120,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )

    return result[0]['generated_text']

question = st.text_input("Ask Einstein:")

if st.button("Ask"):
    answer = chat(question)
    st.write(answer)

    tts = gTTS(answer, lang='ar')
    tts.save("voice.mp3")
    st.audio("voice.mp3")
