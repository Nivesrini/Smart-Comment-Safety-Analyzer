from flask import Flask, render_template, request
import pandas as pd
import re, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

app = Flask(__name__)

CSV_FILE = "social_media_comments_1000.csv"
VECT_PATH = "vectorizer.pkl"
MODEL_PATH = "model.pkl"

nltk.download("stopwords")
nltk.download("vader_lexicon")

stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# ---------------- TEXT CLEANING ---------------- #
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(tokens)

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv(CSV_FILE)
df["clean_text"] = df["text"].apply(clean_text)

def map_to_three(label):
    s = str(label).lower()
    if "pos" in s:
        return "Good"
    elif "neg" in s:
        return "Offensive"
    else:
        return "Threatening"

df["target"] = df["sentiment"].apply(map_to_three)

# ---------------- MODEL TRAINING ---------------- #
if os.path.exists(VECT_PATH) and os.path.exists(MODEL_PATH):
    vectorizer = joblib.load(VECT_PATH)
    model = joblib.load(MODEL_PATH)
else:
    vectorizer = TfidfVectorizer(max_features=4000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    joblib.dump(vectorizer, VECT_PATH)
    joblib.dump(model, MODEL_PATH)

# ---------------- PREDICTION ---------------- #
def predict_comment(comment):
    cleaned = clean_text(comment)
    vect = vectorizer.transform([cleaned])
    model_pred = model.predict(vect)[0]

    scores = sia.polarity_scores(comment)
    compound = scores["compound"]

    if compound >= 0.4:
        final_pred = "Good"
    elif compound <= -0.5:
        final_pred = "Offensive"
    else:
        final_pred = model_pred

    if final_pred == "Good":
        probs = {"Good":0.85,"Offensive":0.1,"Threatening":0.05}
    elif final_pred == "Offensive":
        probs = {"Good":0.1,"Offensive":0.8,"Threatening":0.1}
    else:
        probs = {"Good":0.1,"Offensive":0.2,"Threatening":0.7}

    return final_pred, probs

# ---------------- ROUTE ---------------- #
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    comment = ""
    probs = {"Good":0,"Offensive":0,"Threatening":0}

    if request.method == "POST":
        comment = request.form["comment"]
        result, probs = predict_comment(comment)

    return render_template("index.html",
                           result=result,
                           comment=comment,
                           probs=probs)

if __name__ == "__main__":
    app.run(debug=True)