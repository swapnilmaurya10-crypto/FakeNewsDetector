import streamlit as st
import re
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔥 IMPORT MODEL
from model import train_model

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Fake News Detector", layout="centered")

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def load_model():
    return train_model()

with st.spinner("Initializing model…"):
    model, vectorizer, accuracy, cm = load_model()

# ---------------- FUNCTIONS ----------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

def predict_news(news):
    news_vector = vectorizer.transform([clean_text(news)])
    prediction = model.predict(news_vector)[0]
    prob = model.predict_proba(news_vector).max()
    return prediction, round(prob * 100, 2)

def rule_based_fact_check(text):
    text = text.lower()
    facts = {
        "narendra modi": "india",
        "joe biden": "usa",
        "vladimir putin": "russia",
        "rishi sunak": "uk"
    }
    for person, country in facts.items():
        if person in text:
            return "MATCH" if country in text else "MISMATCH"
    return "UNKNOWN"

# 🔥 IMPROVED FACT CHECK (REAL FIX)
def fact_check_news_sources(text):
    try:
        query = text[:80]
        results = list(search(query, num_results=5))

        similarities = []

        for url in results:
            try:
                response = requests.get(url, timeout=3)
                soup = BeautifulSoup(response.text, "html.parser")

                title = soup.title.string if soup.title else ""
                if not title:
                    continue

                vect = TfidfVectorizer().fit_transform([text, title])
                sim = cosine_similarity(vect[0:1], vect[1:2])[0][0]

                similarities.append(sim)

            except:
                continue

        if len(similarities) == 0:
            return "UNKNOWN"

        max_sim = max(similarities)

        if max_sim > 0.4:
            return "MATCH"
        else:
            return "MISMATCH"

    except:
        return "UNKNOWN"

# ---------------- LOGIN PAGE ----------------
def login_page():
    st.markdown("""
    <div class="login-card">
        <div class="eyebrow">Restricted Access</div>
        <h1 class="pg-title">Verify<br><em>Your Identity</em></h1>
        <p class="pg-sub">AI-powered misinformation analysis system</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    if st.button("Authenticate & Enter"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

# ---------------- MAIN APP ----------------
def main_app():
    _, col_btn = st.columns([6, 1])
    with col_btn:
        if st.button("Sign Out"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("""
    <div class="eyebrow">AI-Powered Detection</div>
    <h1 class="pg-title">Is this news<br><em>real or fabricated?</em></h1>
    <p class="pg-sub">Hybrid ML + Smart Web Verification</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sec-label">News Input</div>', unsafe_allow_html=True)
    user_input = st.text_area("", placeholder="Paste a news article or headline here…", height=160)

    if st.button("🔎  Analyze News"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter news text.")
        else:
            result, confidence = predict_news(user_input)
            rule_result = rule_based_fact_check(user_input)
            fact_result = fact_check_news_sources(user_input)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            # 🔥 FINAL DECISION LOGIC
            if rule_result == "MISMATCH":
                st.error("🔴 FAKE — Logical mismatch detected")

            elif fact_result == "MATCH":
                if confidence >= 60:
                    st.success(f"🟢 REAL — Verified from news sources ({confidence}%)")
                else:
                    st.warning("🟡 Verified but low confidence")

            elif fact_result == "MISMATCH":
                st.error("🔴 FAKE — Not matching real news articles")

            else:
                st.warning("🟡 UNCERTAIN — Could not verify")

            st.markdown(
                f'<div class="pred-row">Prediction: <span>{result}</span>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Confidence: <span>{confidence}%</span></div>',
                unsafe_allow_html=True,
            )

            st.progress(confidence / 100)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sec-label">Instructions & Guide</div>', unsafe_allow_html=True)
    with st.expander("ℹ️  How to interpret results"):
        st.markdown("""
- REAL → Verified from real news + ML confidence  
- FAKE → Logical mismatch or no matching article  
- UNCERTAIN → Could not verify  
""")

    st.markdown('<div class="sec-label" style="margin-top:1.5rem">Model Performance</div>', unsafe_allow_html=True)
    st.metric("Accuracy", f"{round(accuracy * 100, 2)}%")

# ---------------- ROUTING ----------------
if st.session_state.logged_in:
    main_app()
else:
    login_page()