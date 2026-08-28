"""
train_model.py
==============
Sentiment Analysis Dashboard — Capstone Project (Task 28)
Part B steps: Data Preparation, Model Building & Evaluation

Pipeline:
1. Load data from data/reviews_dataset.csv
2. Clean the text (normalize, remove duplicate whitespace)
3. Split the data into train / validation / test (70/15/15), stratified,
   to keep the proportion of each class equal across all sets
4. Vectorize the text with TF-IDF (instead of static/contextual embeddings
   as compared in Task 27 — TF-IDF + Logistic Regression was chosen because
   the deployment environment has no connection to the Hugging Face Hub, and
   a fast, small, inference-friendly model suited to CPU/serverless was needed)
5. Train a Logistic Regression model (multi-class) with initial hyperparameters
6. Evaluate with accuracy / precision / recall / F1 on the test set
7. Run a second iteration, tuning hyperparameters (C, ngram_range) and compare before/after
8. Save the best model and vectorizer as model/sentiment_model.joblib
"""
import json
import re
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/reviews_dataset.csv"
MODEL_OUT = "model/sentiment_model.joblib"
METRICS_OUT = "model/metrics.json"


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )
    return train_df, val_df, test_df


def evaluate(model, vectorizer, df):
    X = vectorizer.transform(df["text"])
    y_true = df["label"]
    y_pred = model.predict(X)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def train_once(train_df, val_df, test_df, ngram_range, C):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    start = time.time()
    model = LogisticRegression(max_iter=1000, C=C)
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 3)

    val_metrics = evaluate(model, vectorizer, val_df)
    test_metrics = evaluate(model, vectorizer, test_df)
    return model, vectorizer, val_metrics, test_metrics, train_time


def main():
    train_df, val_df, test_df = load_and_prepare()
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    print("\n=== Iteration 1 (baseline): ngram_range=(1,1), C=1.0 ===")
    model1, vec1, val1, test1, t1 = train_once(train_df, val_df, test_df, (1, 1), 1.0)
    print(f"Val F1: {val1['f1_macro']}  Test F1: {test1['f1_macro']}  (train time {t1}s)")

    print("\n=== Iteration 2 (tuned): ngram_range=(1,2), C=5.0 ===")
    model2, vec2, val2, test2, t2 = train_once(train_df, val_df, test_df, (1, 2), 5.0)
    print(f"Val F1: {val2['f1_macro']}  Test F1: {test2['f1_macro']}  (train time {t2}s)")

    if val2["f1_macro"] >= val1["f1_macro"]:
        best_model, best_vec, best_test, chosen = model2, vec2, test2, "iteration_2"
    else:
        best_model, best_vec, best_test, chosen = model1, vec1, test1, "iteration_1"

    print(f"\nSelected model: {chosen}")
    print(f"Final Test metrics: accuracy={best_test['accuracy']} f1_macro={best_test['f1_macro']}")

    joblib.dump(
        {
            "model": best_model,
            "vectorizer": best_vec,
            "labels": sorted(train_df["label"].unique().tolist()),
        },
        MODEL_OUT,
    )

    comparison = {
        "iteration_1_baseline": {"val_f1": val1["f1_macro"], "test_f1": test1["f1_macro"], "test_accuracy": test1["accuracy"]},
        "iteration_2_tuned": {"val_f1": val2["f1_macro"], "test_f1": test2["f1_macro"], "test_accuracy": test2["accuracy"]},
        "selected": chosen,
        "final_test_metrics": {
            "accuracy": best_test["accuracy"],
            "precision_macro": best_test["precision_macro"],
            "recall_macro": best_test["recall_macro"],
            "f1_macro": best_test["f1_macro"],
        },
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved metrics -> {METRICS_OUT}")

    X_test = best_vec.transform(test_df["text"])
    preds = best_model.predict(X_test)
    errors = test_df.assign(pred=preds)
    errors = errors[errors["label"] != errors["pred"]]
    print(f"\nError analysis: {len(errors)} / {len(test_df)} misclassified on test set")
    for _, row in errors.head(5).iterrows():
        print(f"  text={row['text'][:60]!r} true={row['label']} pred={row['pred']}")


if __name__ == "__main__":
    main()
