import streamlit as st
import joblib
import os
import numpy as np

MODEL_DIR = "model"
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
CLF_PATH = os.path.join(MODEL_DIR, "classifier.joblib")

def load_models():
    vec = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
    return vec, clf

st.set_page_config(page_title="Career Recommender", layout="centered")
st.title("AI Career Recommendation System")
st.write("Enter your skills (comma separated).")

skills_input = st.text_area("Your skills (e.g. python, sql, pandas, machine learning)", height=120)

if st.button("Recommend careers"):
    if not skills_input.strip():
        st.error("Please enter at least one skill.")
    else:
        vec, clf = load_models()
        Xv = vec.transform([skills_input])
        proba = clf.predict_proba(Xv)[0]
        classes = clf.classes_
        idx = np.argsort(proba)[::-1][:5]
        st.subheader("Top recommendations")
        for i in idx:
            st.markdown(f"**{classes[i]}** — confidence: {proba[i]:.2f}")
