import os
import streamlit as st
import openai
import pandas as pd
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Intelligent Chatbot", layout="wide")
st.title("Intelligent Chatbot — OpenAI + Local FAQ fallback")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

faq_path = "data/faqs.csv"
if os.path.exists(faq_path):
    faqs = pd.read_csv(faq_path)
    questions = faqs["question"].fillna("").tolist()
else:
    faqs = pd.DataFrame(columns=["question","answer"])
    questions = []

if "messages" not in st.session_state:
    st.session_state.messages = []

def call_openai(prompt):
    openai.api_key = OPENAI_KEY
    messages = [{"role":"system","content":"You are a helpful assistant."}]
    for m in st.session_state.messages:
        messages.append(m)
    messages.append({"role":"user","content":prompt})
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.2
        )
        reply = resp["choices"][0]["message"]["content"].strip()
        return reply
    except Exception as e:
        return "OpenAI request failed: " + str(e)

def local_fallback(query):
    if not questions:
        return "No FAQ data found."
    best = process.extractOne(query, questions, scorer=fuzz.ratio)
    if best and best[1] > 60:
        idx = questions.index(best[0])
        return faqs.iloc[idx]["answer"]
    else:
        return "I couldn't find a close FAQ match. Try rephrasing or enable OpenAI key."

user_input = st.text_area("Your question", height=130)
if st.button("Send"):
    if not user_input.strip():
        st.warning("Please type a question.")
    else:
        if OPENAI_KEY:
            st.session_state.messages.append({"role":"user","content":user_input})
            reply = call_openai(user_input)
            st.session_state.messages.append({"role":"assistant","content":reply})
        else:
            reply = local_fallback(user_input)
            st.session_state.messages.append({"role":"assistant","content":reply})

for msg in st.session_state.messages[::-1]:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**Bot:** {content}")
