# Arabic Customer Reviews — Sentiment Analysis

A complete Arabic Natural Language Processing (NLP) pipeline that classifies customer reviews written in Arabic as **positive**, **neutral**, or **negative**. Built as part of the AI & Data Science Internship at ACUD (Administrative Capital for Urban Development).

## Overview

This project takes real customer reviews written in Arabic (collected from multiple companies such as Talabat, Swvl, Telecom Egypt, and others), cleans and normalizes the text, and trains a machine learning model to predict the sentiment of any new review. The trained model is deployed as an interactive Streamlit app where a user can type a review in Arabic and instantly get a sentiment prediction with a confidence score per class.

## Features

- Arabic-specific text cleaning: removes diacritics, normalizes alef/alef maqsura/ta marbuta variants, strips numbers and non-Arabic characters
- Negation-aware stopword removal — keeps important negation words (e.g. مش, لا, ما, مفيش, ليس, لم, لن) instead of dropping them like a standard stopword list would, so the model doesn't lose the meaning of negated sentences
- TF-IDF vectorization with unigrams and bigrams
- Logistic Regression classifier for 3-class sentiment classification (positive / neutral / negative)
- Exploratory data analysis: sentiment distribution, per-company breakdown, review length analysis, top words per sentiment class
- Full evaluation: accuracy, classification report, confusion matrix
- Interactive Streamlit web app for live predictions with per-class confidence chart
- Option to use the default dataset or upload a custom CSV file with the same format

## Dataset

- **File:** `Final_Data.csv`
- **Columns:** `review_description`, `rating` (sentiment label), `company`
- ~40,000 Arabic customer reviews collected across multiple companies, covering a mix of positive, neutral, and negative sentiment

## Tech Stack

- **Language:** Python
- **NLP:** NLTK (Arabic stopwords), custom Arabic text normalization
- **Machine Learning:** scikit-learn (TF-IDF, Logistic Regression)
- **Data Analysis & Visualization:** pandas, NumPy, matplotlib, seaborn
- **Web App:** Streamlit

## Project Structure

```
customer reviews in Arabic/
├── app.py               # Streamlit web application
├── code.ipynb            # Full analysis and model training notebook
├── Final_Data.csv         # Arabic customer reviews dataset
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

You have two options: run the app yourself from the terminal, or just open it directly from the live link with no setup at all.

```bash
git clone https://github.com/OmarAhmedRamadan07/customer_reviews_in_Arabic.git
cd customer_reviews_in_Arabic
pip install -r requirements.txt
```

## Usage

### Option 1 — Run the web app from the terminal

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, write a review in Arabic, and click **Evaluate**.

### Option 2 — Open it directly from the link (no source code needed)

You don't have to clone the repo or install anything at all. The app is already deployed and ready to use straight from your browser:

https://customer-reviews-inarabic.streamlit.app/

### Run the notebook

Open `code.ipynb` to walk through the full pipeline: data cleaning, exploratory analysis, feature engineering, model training, and evaluation.

## How It Works

1. **Data Cleaning** — missing values are dropped, duplicate reviews are removed, and the `rating` column (which actually holds the sentiment label) is renamed to `sentiment`.
2. **Text Preprocessing** — Arabic diacritics are stripped, letter variants are normalized (إ/أ/آ → ا, ى → ي, ة → ه), numbers and non-Arabic characters are removed, and stopwords are filtered out while preserving key negation words.
3. **Feature Extraction** — the cleaned text is converted into numerical features using TF-IDF with both unigrams and bigrams (top 5,000 features).
4. **Model Training** — a Logistic Regression classifier is trained on an 80/20 stratified train-test split.
5. **Evaluation** — model performance is measured with accuracy, a full classification report, and a confusion matrix; the top words driving each sentiment class are also inspected.
6. **Prediction** — new reviews go through the exact same cleaning and vectorization pipeline before the trained model predicts their sentiment and confidence scores.

## Notes

- The Streamlit app retrains the model once per session (cached) from `Final_Data.csv`, so no pre-saved model file is needed.
- Sentiment analysis is currently 3-class: positive, neutral, negative.
- Works with Arabic text only; the cleaning pipeline strips any non-Arabic characters.

## Acknowledgments

Built as part of the AI & Data Science Internship at ACUD (Administrative Capital for Urban Development), CET191 — Internship I, El Sewedy University of Technology.

## License

This project is provided for educational and portfolio purposes.
