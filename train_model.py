import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = "data/careers.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df["skills"].fillna("")
y = df["career"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=200))
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
print("Classification report:")
print(classification_report(y_test, preds))

joblib.dump(pipe.named_steps["tfidf"], os.path.join(MODEL_DIR, "vectorizer.joblib"))
joblib.dump(pipe.named_steps["clf"], os.path.join(MODEL_DIR, "classifier.joblib"))
print("Saved vectorizer and classifier in", MODEL_DIR)
