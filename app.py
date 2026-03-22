import streamlit as st
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia

st.set_page_config(page_title="AI Fake News Detector", layout="centered")


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
accuracy = joblib.load("accuracy.pkl")
cm = joblib.load("cm.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def predict_news(news):
    news = clean_text(news)
    news_vector = vectorizer.transform([news])
    
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
            if country not in text:
                return "MISMATCH"
            else:
                return "MATCH"

    return "UNKNOWN"


def fact_check_wikipedia(text):
    try:
        results = wikipedia.search(text)

        if len(results) == 0:
            return "UNKNOWN"

        summary = wikipedia.summary(results[0], sentences=2)

        text_words = set(text.lower().split())
        summary_words = set(summary.lower().split())

        common_words = text_words.intersection(summary_words)

        if len(common_words) / len(text_words) > 0.4:
            return "MATCH"
        else:
            return "MISMATCH"

    except:
        return "UNKNOWN"


st.title("🧠 AI Fake News Detector")
st.caption("Analyze any news article using Machine Learning + Fact Checking")

user_input = st.text_area("📰 Paste News Article or Headline")


if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter news text.")

    else:
        with st.spinner("Analyzing with AI + Fact Check..."):
            result, confidence = predict_news(user_input)
            rule_result = rule_based_fact_check(user_input)

        

        
        if rule_result == "MISMATCH":
            st.error("🔴 FAKE (Logical fact mismatch detected)")

        else:
            fact_result = fact_check_wikipedia(user_input)

          
            if fact_result == "MISMATCH":
                st.error("🔴 FAKE (Fact-check mismatch detected)")

            
            elif fact_result == "UNKNOWN":
                if confidence < 75:
                    st.warning(f"🟡 UNCERTAIN ({confidence}% confidence)")
                else:
                    st.info("🟡 Could not verify from knowledge base")

            
            else:
                if result == "REAL":
                    st.success(f"🟢 REAL News ({confidence}% confidence)")
                else:
                    st.error(f"🔴 FAKE News ({confidence}% confidence)")

        
        st.write(f"ML Prediction: {result}")
        st.write(f"Confidence: {confidence}%")

        
        st.progress(confidence / 100)

st.subheader("📊 Model Performance")
st.write(f"Accuracy: {round(accuracy*100,2)}%")


st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["FAKE", "REAL"],
            yticklabels=["FAKE", "REAL"])

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)