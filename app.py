import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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
# اختيار الشخصية
persona = st.selectbox(
    "اختر الشخصية:",
    [
        "ألبرت أينشتاين",
        "نيوتن",
        "صلاح الدين الأيوبي",
        "أرسطو",
        "ابن سينا"
    ]
)

question = st.text_input("اكتب سؤالك:")

def ask(persona, question):
    if not question.strip():
        return "من فضلك اكتب سؤال."

    prompt = f"""
أنت تتحدث بصوت شخصية تاريخية اسمها {persona}.
أجب باللغة العربية الفصحى فقط.
كن دقيقًا ومبسطًا ولا تخترع معلومات غير صحيحة.

السؤال: {question}
الإجابة:
"""

    result = model(
        prompt,
        max_new_tokens=128
    )

    return result[0]["generated_text"]

if st.button("اسأل"):
    answer = ask(persona, question)
    st.write("🧠 الإجابة:")
    st.success(answer)
