import streamlit as st
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Fake News Detector", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #312e81, #0f172a);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
    color: white;
}
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    filter: blur(120px);
    z-index: -1;
    animation: float 10s ease-in-out infinite;
}
.stApp::before {
    background: #6366f1;
    top: 10%;
    left: 10%;
}
.stApp::after {
    background: #8b5cf6;
    bottom: 10%;
    right: 10%;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-30px); }
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: bold;
}
.result-box {
    padding: 15px;
    border-radius: 14px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}
.stExpander {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
accuracy = joblib.load("accuracy.pkl")
cm = joblib.load("cm.pkl")

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

def fact_check_wikipedia(text):
    try:
        results = wikipedia.search(text)
        if not results:
            return "UNKNOWN"
        summary = wikipedia.summary(results[0], sentences=2)
        text_words = set(text.lower().split())
        summary_words = set(summary.lower().split())
        return "MATCH" if len(text_words & summary_words)/len(text_words) > 0.4 else "MISMATCH"
    except:
        return "UNKNOWN"

# ---------------- LOGIN PAGE ----------------
def login_page():
    st.markdown("<h1>🔐 Login</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Access AI Fake News Detector</p>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login Successful")
            st.rerun()
        else:
            st.error("❌ Invalid Credentials")

# ---------------- MAIN APP ----------------
def main_app():

    # Logout
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("<h1>🧠 AI Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Hybrid ML + Fact Checking System</p>", unsafe_allow_html=True)

    # INPUT
    user_input = st.text_area("📰 Enter News", placeholder="Paste news here...")

    # ANALYZE
    if st.button("🚀 Analyze News"):

        if user_input.strip() == "":
            st.warning("⚠️ Please enter news text.")
        else:
            result, confidence = predict_news(user_input)
            rule_result = rule_based_fact_check(user_input)

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            if rule_result == "MISMATCH":
                st.error("🔴 FAKE (Logical mismatch detected)")
            else:
                fact_result = fact_check_wikipedia(user_input)

                if fact_result == "MISMATCH":
                    st.error("🔴 FAKE (Fact-check mismatch detected)")
                elif fact_result == "UNKNOWN":
                    if confidence < 75:
                        st.warning(f"🟡 UNCERTAIN ({confidence}%)")
                    else:
                        st.info("🟡 Could not verify")
                else:
                    if result == "REAL":
                        st.success(f"🟢 REAL ({confidence}%)")
                    else:
                        st.error(f"🔴 FAKE ({confidence}%)")

            st.write(f"Prediction: {result} | Confidence: {confidence}%")
            st.progress(confidence / 100)
            st.markdown("</div>", unsafe_allow_html=True)

    # INSTRUCTIONS (FIXED POSITION)
    st.markdown("### 📘 Instructions & Interpretation Guide")

    with st.expander("ℹ️ Click to view instructions"):
        st.markdown("""
### 🧠 How to Use
- Paste any news headline or article  
- Click **Analyze News**

### 📊 Interpretation
- 🟢 REAL → Confidence ≥ 75%  
- 🔴 FAKE → Logical mismatch / fact mismatch  
- 🟡 UNCERTAIN → Confidence < 75%

### ⚠️ Note
- Uses ML + Rule-based + Wikipedia  
- Detects patterns, not absolute truth  
""")

    # PERFORMANCE
    st.markdown("### 📊 Model Performance")
    st.metric("Accuracy", f"{round(accuracy*100,2)}%")

# ---------------- ROUTING ----------------
if st.session_state.logged_in:
    main_app()
else:
    login_page()