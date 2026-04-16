import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.title("🧠 محادثة مع الشخصيات التاريخية")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

# ✅ مهم جدًا: تعريف model هنا
model = load_model()

persona = st.selectbox(
    "اختر الشخصية:",
    ["أينشتاين", "نيوتن", "ابن سينا", "صلاح الدين الأيوبي"]
)

question = st.text_input("اكتب سؤالك:")

def ask(persona, question):
    if not question.strip():
        return "اكتب سؤال من فضلك"

    prompt = f"""
أنت تمثل شخصية تاريخية اسمها {persona}.
أجب باللغة العربية الفصحى بشكل دقيق ومختصر.

السؤال: {question}
الإجابة:
"""

    result = model(prompt, max_new_tokens=128)
    return result[0]["generated_text"]

if st.button("اسأل"):
    answer = ask(persona, question)
    st.success(answer)
