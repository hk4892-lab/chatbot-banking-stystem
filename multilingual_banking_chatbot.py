#!/usr/bin/env python3
# Multilingual Banking Chatbot using Machine Learning (English–Tamil)
# Allowed libraries: pandas, sklearn, matplotlib, seaborn, pickle, re (plus Python stdlib)

import re
import pickle

import matplotlib
matplotlib.use("Agg")  # headless backend for servers/CI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# ----------------------------
# Utility: Column standardization
# ----------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=canonical_map)

    user_candidates = [
        "user", "question", "query", "utterance", "text", "customer_query",
    ]
    bot_candidates = [
        "bot", "answer", "response", "reply", "bot_response",
    ]
    intent_candidates = [
        "intent", "label", "tag", "category", "class",
    ]

    def pick(col_list):
        for c in col_list:
            if c in df.columns:
                return c
        return None

    user_col = pick(user_candidates)
    bot_col = pick(bot_candidates)
    intent_col = pick(intent_candidates)

    missing = [name for name, col in [("user", user_col), ("bot", bot_col), ("intent", intent_col)] if col is None]
    if missing:
        raise ValueError(
            f"Could not find required columns {missing} in CSV. Available columns: {list(df.columns)}"
        )

    return df[[user_col, bot_col, intent_col]].rename(columns={user_col: "user", bot_col: "bot", intent_col: "intent"})


# ----------------------------
# Step 3: Preprocessing
# ----------------------------

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Keep word chars (includes digits/underscore), whitespace, and Tamil block
    text = re.sub(r"[^\w\s\u0B80-\u0BFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    return "Tamil" if re.search(r"[\u0B80-\u0BFF]", str(text)) else "English"


# ----------------------------
# Chatbot helper
# ----------------------------

def chatbot_response(user_input: str, df: pd.DataFrame, model: LogisticRegression, vectorizer: TfidfVectorizer) -> str:
    lang = detect_language(user_input)
    cleaned = clean_text(user_input)
    pred_intent = model.predict(vectorizer.transform([cleaned]))[0]

    # Prefer reply in same language; fallback to any reply with same intent
    subset = df[(df["intent"] == pred_intent) & (df["lang"] == lang)]
    if subset.empty:
        subset = df[df["intent"] == pred_intent]

    if not subset.empty:
        reply = subset.sample(n=1, random_state=42).iloc[0]["bot"]
    else:
        reply = "I'm here to help with your banking query."

    return f"[{lang}] [Intent: {pred_intent}] -> {reply}"


# ----------------------------
# Main pipeline
# ----------------------------

def main():
    # Step 2: Load Datasets
    path_eng = "/mnt/data/BankFAQs.csv"
    path_tam = "/mnt/data/BankFAQs_Tamil.csv"

    try:
        eng_df_raw = pd.read_csv(path_eng)
        tam_df_raw = pd.read_csv(path_tam)
    except FileNotFoundError as e:
        print("[ERROR] One or both dataset files not found.")
        print(f"Expected: {path_eng} and {path_tam}")
        print("Please place the CSV files at the specified paths and re-run.")
        raise SystemExit(1)
    except Exception as e:
        print("[ERROR] Failed to load datasets:", e)
        raise SystemExit(1)

    try:
        eng_df = standardize_columns(eng_df_raw)
        tam_df = standardize_columns(tam_df_raw)
    except Exception as e:
        print("[ERROR] Column standardization failed:", e)
        raise SystemExit(1)

    # Step 1/2: Merge and auto-add language column
    df = pd.concat([eng_df, tam_df], ignore_index=True)
    df["lang"] = df["user"].apply(detect_language)

    print(f"✅ Combined dataset shape: {df.shape}")
    print(df.head())

    # Step 3: Preprocessing
    df["clean_text"] = df["user"].apply(clean_text)

    # Step 4: Train-Test Split
    X = df["clean_text"].values
    y = df["intent"].values

    # Use stratify if all classes have at least 2 samples
    class_counts = pd.Series(y).value_counts()
    stratify_val = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_val
    )

    # Step 5: Vectorize + Train Model
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False)
    Xv_train = vectorizer.fit_transform(X_train)
    Xv_test = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(Xv_train, y_train)
    preds = model.predict(Xv_test)

    # Step 6: Evaluation
    report = classification_report(y_test, preds, digits=4)
    print("\n--- Classification Report ---\n", report)

    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("Multilingual Banking Chatbot - Intent Classification Report\n")
        f.write("==============================================\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n\n")
        f.write(report)

    # Confusion matrix
    classes = list(model.classes_)
    cm = confusion_matrix(y_test, preds, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pd.DataFrame(cm, index=classes, columns=classes),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("Confusion Matrix - Intent Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # Intent distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x="intent", data=df, order=df["intent"].value_counts().index)
    plt.title("Intent Distribution")
    plt.xlabel("Intent")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("intent_distribution.png", dpi=150)
    plt.close()

    # Step 7: Chatbot functions (demo)
    english_queries = [
        "I want to open an account",
        "Check my loan status",
        "Lost my credit card",
        "Update KYC details",
        "What is the FD interest rate?",
    ]
    tamil_queries = [
        "எனக்கு ஒரு புதிய கணக்கு திறக்க வேண்டும்",
        "என் கடன் நிலை என்ன?",
        "என் கார்டு தொலைந்துவிட்டது",
        "எனது KYC புதுப்பிக்க வேண்டும்",
        "நிலுவை வட்டி விகிதம் என்ன?",
    ]

    print("\n🤖 Sample English Queries:")
    for q in english_queries:
        print("\nUser:", q)
        print("Bot :", chatbot_response(q, df, model, vectorizer))

    print("\n🤖 Sample Tamil Queries:")
    for q in tamil_queries:
        print("\nUser:", q)
        print("Bot :", chatbot_response(q, df, model, vectorizer))

    # Step 9: Save models
    with open("intent_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n✅ All files saved: intent_model.pkl, vectorizer.pkl, report.txt, confusion_matrix.png, intent_distribution.png")


if __name__ == "__main__":
    main()
