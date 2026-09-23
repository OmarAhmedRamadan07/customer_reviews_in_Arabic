import re
import pandas as pd
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Arabic Review Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
)

# ============================================================
# NLTK SETUP
# ============================================================
@st.cache_resource
def download_nltk_data():
    try:
        stopwords.words("arabic")
    except LookupError:
        nltk.download("stopwords")


download_nltk_data()

IMPORTANT_NEGATIONS = {"مش", "موش", "لا", "ما", "مفيش", "ليس", "لم", "لن"}


def get_arabic_stopwords():
    sw = set(stopwords.words("arabic"))
    sw -= IMPORTANT_NEGATIONS
    return sw


# ============================================================
# TEXT CLEANING (same pipeline as the training notebook)
# ============================================================
def clean_arabic_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)      # remove diacritics
    text = re.sub(r"[إأآا]", "ا", text)                     # normalize alef
    text = re.sub(r"ى", "ي", text)                          # normalize alef maqsura
    text = re.sub(r"ة", "ه", text)                          # normalize ta marbuta
    text = re.sub(r"\d+", " ", text)                        # remove numbers
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)          # keep Arabic chars only
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_stopwords(text: str, arabic_stopwords: set) -> str:
    words = text.split()
    filtered = [w for w in words if w not in arabic_stopwords]
    return " ".join(filtered)


def preprocess(text: str, arabic_stopwords: set) -> str:
    cleaned = clean_arabic_text(text)
    return remove_stopwords(cleaned, arabic_stopwords)


# ============================================================
# MODEL TRAINING (cached so it only runs once per session)
# ============================================================
@st.cache_resource(show_spinner="Training the sentiment model...")
def train_model(csv_path: str):
    df = pd.read_csv(csv_path)

    df.dropna(subset=["review_description"], inplace=True)
    df.dropna(subset=["rating"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"rating": "sentiment"}, inplace=True)

    arabic_stopwords = get_arabic_stopwords()

    df["cleaned_review"] = df["review_description"].apply(clean_arabic_text)
    df["review_no_stopwords"] = df["cleaned_review"].apply(
        lambda t: remove_stopwords(t, arabic_stopwords)
    )

    df.drop_duplicates(subset=["review_description"], inplace=True)
    df = df[df["review_no_stopwords"].str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)

    X = df["review_no_stopwords"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))

    return model, tfidf, arabic_stopwords, accuracy, len(df)


# ============================================================
# UI
# ============================================================
st.title("💬 Arabic Review Sentiment Analyzer")
st.write(
    "Type a review below in Arabic and the model will predict whether it's "
    "**positive**, **neutral**, or **negative**."
)

with st.sidebar:
    st.header("Dataset")
    data_source = st.radio(
        "Choose data source",
        ["Use default path (Final_Data.csv)", "Upload a CSV file"],
    )

    csv_path = None
    if data_source == "Use default path (Final_Data.csv)":
        csv_path = "Final_Data.csv"
    else:
        uploaded_file = st.file_uploader("Upload Final_Data.csv", type=["csv"])
        if uploaded_file is not None:
            csv_path = uploaded_file

model_ready = False
if csv_path is not None:
    try:
        model, tfidf, arabic_stopwords, accuracy, n_rows = train_model(csv_path)
        model_ready = True
        st.sidebar.success(f"Model trained on {n_rows} reviews")
        st.sidebar.metric("Test accuracy", f"{accuracy:.2%}")
    except FileNotFoundError:
        st.sidebar.error(
            "Final_Data.csv was not found next to app.py. "
            "Place the file there, or upload it from the sidebar."
        )
    except Exception as e:
        st.sidebar.error(f"Failed to train the model: {e}")

st.divider()

user_text = st.text_area(
    "Write your review here",
    height=140,
    placeholder="اكتب رأيك هنا...",
)

col1, col2 = st.columns([1, 3])
with col1:
    evaluate_clicked = st.button("Evaluate", type="primary", use_container_width=True)

if evaluate_clicked:
    if not model_ready:
        st.warning("The model isn't ready yet — check the dataset in the sidebar.")
    elif not user_text.strip():
        st.warning("Please write something first.")
    else:
        processed = preprocess(user_text, arabic_stopwords)
        vec = tfidf.transform([processed])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        classes = model.classes_

        label_style = {
            "positive": ("🟢 Positive", "success"),
            "neutral": ("🟡 Neutral", "info"),
            "negative": ("🔴 Negative", "error"),
        }
        label_text, style = label_style.get(prediction, (prediction, "info"))

        getattr(st, style)(f"Predicted sentiment: **{label_text}**")

        st.write("Confidence per class:")
        prob_df = pd.DataFrame({"sentiment": classes, "probability": probs})
        prob_df = prob_df.set_index("sentiment")
        st.bar_chart(prob_df)
