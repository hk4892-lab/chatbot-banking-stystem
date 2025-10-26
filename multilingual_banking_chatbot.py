#!/usr/bin/env python3
# Multilingual Banking Chatbot using Machine Learning (English–Tamil)
# Allowed libraries only: pandas, sklearn, matplotlib, seaborn, pickle, re

import re
import pickle
import warnings

import pandas as pd

import matplotlib
matplotlib.use('Agg')  # headless backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------
# Utility functions
# ------------------------------

def normalize_col(col_name: str) -> str:
    return re.sub(r"[^a-z]", "", str(col_name).lower())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map dataset columns to expected: 'user', 'bot', 'intent'.
    Tries common synonyms by normalizing column names.
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe provided for standardization.")

    normalized_to_original = {normalize_col(c): c for c in df.columns}

    synonyms = {
        "user": {"user", "question", "query", "userquery", "customerquestion", "input", "utterance"},
        "bot": {"bot", "answer", "response", "botresponse", "reply", "botreply"},
        "intent": {"intent", "label", "category", "class", "tag"},
    }

    mapping = {}
    for target, choices in synonyms.items():
        found = None
        for norm, orig in normalized_to_original.items():
            if norm in choices:
                found = orig
                break
        if not found:
            # try exact expected names as a final fallback
            if target in df.columns:
                found = target
            else:
                raise ValueError(f"Could not find a column for '{target}' in columns: {list(df.columns)}")
        mapping[found] = target

    return df.rename(columns=mapping)[["user", "bot", "intent"]]


def detect_language(text: str) -> str:
    if pd.isna(text):
        return "English"
    return "Tamil" if re.search(r"[\u0B80-\u0BFF]", str(text)) else "English"


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # keep English word chars, spaces, and Tamil Unicode range
    text = re.sub(r"[^\w\s\u0B80-\u0BFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def try_read_csv(paths):
    """Attempt to read CSV from a sequence of paths. Return first successful DataFrame, else None."""
    for p in paths:
        try:
            df = pd.read_csv(p)
            print(f"Loaded dataset: {p}")
            return df
        except Exception as e:
            # Try next path
            continue
    return None


def create_demo_dataset() -> pd.DataFrame:
    """Create a small demo dataset in English and Tamil to allow the pipeline to run."""
    eng_samples = [
        {"user": "I want to open an account", "bot": "To open an account, please provide ID and address proof.", "intent": "open_account"},
        {"user": "Check my loan status", "bot": "You can check your loan status in the loan section.", "intent": "loan_status"},
        {"user": "Lost my credit card", "bot": "We have blocked your card. Please request a replacement.", "intent": "card_lost"},
        {"user": "Update KYC details", "bot": "Visit the nearest branch to update your KYC details.", "intent": "update_kyc"},
        {"user": "What is the FD interest rate?", "bot": "Current FD interest rates are listed on our website.", "intent": "fd_rate"},
        {"user": "How to reset netbanking password", "bot": "Use the Forgot Password option on the login page.", "intent": "reset_password"},
    ]

    tam_samples = [
        {"user": "எனக்கு ஒரு புதிய கணக்கு திறக்க வேண்டும்", "bot": "புதிய கணக்கு திறக்க அடையாள ஆவணம் மற்றும் முகவரி சான்று தேவை.", "intent": "open_account"},
        {"user": "என் கடன் நிலை என்ன?", "bot": "கடன் நிலையை 'Loan' பகுதியில் பார்த்து தெரிந்துகொள்ளலாம்.", "intent": "loan_status"},
        {"user": "என் கார்டு தொலைந்துவிட்டது", "bot": "உங்கள் கார்டு தற்காலிகமாக முடக்கப்பட்டுள்ளது. மாற்று கார்டு விண்ணப்பிக்கவும்.", "intent": "card_lost"},
        {"user": "எனது KYC புதுப்பிக்க வேண்டும்", "bot": "அருகிலுள்ள கிளையை சென்று KYC விவரங்களைப் புதுப்பிக்கவும்.", "intent": "update_kyc"},
        {"user": "நிலுவை வட்டி விகிதம் என்ன?", "bot": "தற்போதைய நிலுவை வட்டி விகிதங்கள் எங்கள் இணையதளத்தில் உள்ளன.", "intent": "fd_rate"},
        {"user": "இணைய வங்கிப் பாஸ்வேர்டை எப்படி மாற்றுவது?", "bot": "Login பக்கத்தில் 'Forgot Password' பயன்படுத்தவும்.", "intent": "reset_password"},
    ]

    eng_df = pd.DataFrame(eng_samples)
    tam_df = pd.DataFrame(tam_samples)
    eng_df["lang"] = "English"
    tam_df["lang"] = "Tamil"
    return pd.concat([eng_df, tam_df], ignore_index=True)


def main():
    print("\n=== Multilingual Banking Chatbot (English–Tamil) ===\n")

    # ------------------------------
    # Step 1-2: Load datasets and merge
    # ------------------------------
    candidate_paths_eng = [
        "/mnt/data/BankFAQs.csv",
        "/workspace/BankFAQs.csv",
        "/workspace/data/BankFAQs.csv",
        "BankFAQs.csv",
        "data/BankFAQs.csv",
    ]
    candidate_paths_tam = [
        "/mnt/data/BankFAQs_Tamil.csv",
        "/workspace/BankFAQs_Tamil.csv",
        "/workspace/data/BankFAQs_Tamil.csv",
        "BankFAQs_Tamil.csv",
        "data/BankFAQs_Tamil.csv",
    ]

    eng_df_raw = try_read_csv(candidate_paths_eng)
    tam_df_raw = try_read_csv(candidate_paths_tam)

    using_demo = False

    if eng_df_raw is not None and tam_df_raw is not None:
        try:
            eng_df = standardize_columns(eng_df_raw)
            tam_df = standardize_columns(tam_df_raw)
            # Auto language detection column
            eng_df["lang"] = eng_df["user"].apply(detect_language)
            tam_df["lang"] = tam_df["user"].apply(detect_language)
            df = pd.concat([eng_df, tam_df], ignore_index=True)
            print(f"✅ Combined dataset shape: {df.shape}")
        except Exception as e:
            print(f"Column standardization failed: {e}\nFalling back to demo dataset.")
            df = create_demo_dataset()
            using_demo = True
    else:
        print("Datasets not found. Using a small demo dataset so the pipeline can run.")
        df = create_demo_dataset()
        using_demo = True

    # ------------------------------
    # Step 3: Preprocessing
    # ------------------------------
    # Clean and normalize text
    df = df.dropna(subset=["user", "bot", "intent"]).copy()
    if "lang" not in df.columns:
        df["lang"] = df["user"].apply(detect_language)
    df["clean_text"] = df["user"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    # ------------------------------
    # Step 4: Train-Test Split
    # ------------------------------
    X = df["clean_text"]
    y = df["intent"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # ------------------------------
    # Step 5: Vectorize + Train Model
    # ------------------------------
    # Use Unicode-aware token pattern; ensure proper backslashes
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    Xv_train = vectorizer.fit_transform(X_train)
    Xv_test = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(Xv_train, y_train)
    preds = model.predict(Xv_test)

    # ------------------------------
    # Step 6: Evaluation
    # ------------------------------
    acc = accuracy_score(y_test, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )

    report = classification_report(y_test, preds, zero_division=0)
    print("\n--- Classification Report ---\n")
    print(report)

    # Save report
    with open("/workspace/report.txt", "w", encoding="utf-8") as f:
        title = "Report for Multilingual Banking Chatbot\n"
        src_info = "Data source: DEMO DATASET (files missing)\n" if using_demo else "Data source: Provided CSV files\n"
        f.write(title)
        f.write(src_info)
        f.write("\nSummary Metrics (macro/weighted):\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision (macro): {precision_macro:.4f} | (weighted): {precision_weighted:.4f}\n")
        f.write(f"Recall (macro): {recall_macro:.4f} | (weighted): {recall_weighted:.4f}\n")
        f.write(f"F1 (macro): {f1_macro:.4f} | (weighted): {f1_weighted:.4f}\n\n")
        f.write(report)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix - Intent Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("/workspace/confusion_matrix.png", dpi=150)
    plt.close()

    # Intent distribution plot
    plt.figure(figsize=(max(8, len(df.intent.unique()) * 1.2), 4))
    df.intent.value_counts().plot(kind="bar", rot=45, color="#4e79a7")
    plt.title("Intent Distribution")
    plt.xlabel("Intent")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("/workspace/intent_distribution.png", dpi=150)
    plt.close()

    # ------------------------------
    # Step 7: Chatbot Functions
    # ------------------------------
    def chatbot_response(user_input: str) -> str:
        lang = detect_language(user_input)
        cleaned = clean_text(user_input)
        pred_intent = model.predict(vectorizer.transform([cleaned]))[0]
        subset = df[df.intent == pred_intent]
        # Prefer same-language reply if available
        preferred = subset[subset.lang == lang]
        if not preferred.empty:
            reply = preferred.sample(1).iloc[0]["bot"]
        elif not subset.empty:
            reply = subset.sample(1).iloc[0]["bot"]
        else:
            reply = "I'm here to help with your banking query."
        return f"[{lang}] [Intent: {pred_intent}] → {reply}"

    # ------------------------------
    # Step 8: Chatbot Demo
    # ------------------------------
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
        print(f"\nUser: {q}")
        print(f"Bot : {chatbot_response(q)}")

    print("\n🤖 Sample Tamil Queries:")
    for q in tamil_queries:
        print(f"\nUser: {q}")
        print(f"Bot : {chatbot_response(q)}")

    # ------------------------------
    # Step 9: Save models
    # ------------------------------
    with open("/workspace/intent_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("/workspace/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n✅ All files saved: intent_model.pkl, vectorizer.pkl, report.txt, confusion_matrix.png, intent_distribution.png")


if __name__ == "__main__":
    main()
