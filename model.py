import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = "FAKE"
real["label"] = "REAL"

data = pd.concat([fake, real])
data = data[["text", "label"]]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data["text"] = data["text"].apply(clean_text)


X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.25, random_state=42
)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Model Accuracy:", accuracy)


joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(accuracy, "accuracy.pkl")
joblib.dump(cm, "cm.pkl")

print("✅ Model saved successfully!")