# ===============================
# EMOTION DETECTION MODEL TRAINING
# Using Sentence and Label Columns
# ===============================

import pandas as pd
import pickle
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# LOAD DATASET
# -------------------------------
# Your CSV must have:
# column: Sentence
# column: Label
# Example labels: happy, sad, angry, fear, neutral

df = pd.read_parquet("emotions_dataset.parquet")

print(df.head())
print(df['Label'].value_counts())

# -------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

df["clean_text"] = df["Sentence"].apply(clean_text)

# -------------------------------
# SPLIT DATA
# -------------------------------
X = df["clean_text"]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# TF-IDF VECTORIZATION
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# SAVE .PKL FILES
# -------------------------------
pickle.dump(model, open("emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("emotion_vectorizer.pkl", "wb"))

print("\n✅ Model saved as emotion_model.pkl")
print("✅ Vectorizer saved as emotion_vectorizer.pkl")
