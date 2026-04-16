import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

st.title("🧠 Einstein AI Chatbot")

generator = pipeline("text-generation", model="gpt2")

def chat(question):
    prompt = f"You are Albert Einstein. Answer wisely: {question}"
    result = generator(prompt, max_length=200)
    return result[0]['generated_text']

question = st.text_input("Ask Einstein:")

if st.button("Ask"):
    answer = chat(question)
    st.write(answer)

    tts = gTTS(answer)
    tts.save("voice.mp3")
    st.audio("voice.mp3")
