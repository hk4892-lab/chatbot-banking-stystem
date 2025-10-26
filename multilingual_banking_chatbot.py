"""
Multilingual Banking Chatbot using Machine Learning (English–Tamil)

This project implements a hybrid chatbot that:
- Detects language (English/Tamil)
- Classifies user intent using TF-IDF + Logistic Regression
- Retrieves appropriate bot responses from the dataset
- Provides comprehensive evaluation metrics and visualizations
"""

# --- Step 1: Imports ---
import pandas as pd
import numpy as np
import re
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("=" * 80)
print("MULTILINGUAL BANKING CHATBOT - English & Tamil")
print("=" * 80)

# --- Step 2: Load Datasets ---
print("\n📂 Loading datasets...")
eng_df = pd.read_csv("BankFAQs.csv")
tam_df = pd.read_csv("BankFAQs_Tamil.csv")

# Add language column
eng_df["lang"] = "English"
tam_df["lang"] = "Tamil"

# Combine datasets
df = pd.concat([eng_df, tam_df], ignore_index=True)
print(f"✅ Combined dataset shape: {df.shape}")
print(f"   - English records: {len(eng_df)}")
print(f"   - Tamil records: {len(tam_df)}")
print(f"   - Total records: {len(df)}")
print(f"\n📊 Dataset columns: {list(df.columns)}")
print(f"\n🎯 Unique intents: {df['intent'].nunique()}")
print(f"   Intents: {sorted(df['intent'].unique())}")
print("\nFirst few records:")
print(df.head())

# --- Step 3: Text Preprocessing ---
print("\n" + "=" * 80)
print("🔧 PREPROCESSING TEXT")
print("=" * 80)

def clean_text(text):
    """
    Clean and normalize text for both English and Tamil.
    - Converts to lowercase
    - Keeps Tamil Unicode characters (U+0B80 to U+0BFF)
    - Removes special characters and extra whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Keep alphanumeric and Tamil Unicode range
    text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_text"] = df["user"].apply(clean_text)
print("✅ Text cleaning completed")
print("\nSample cleaned texts:")
for i in range(3):
    print(f"\nOriginal: {df.iloc[i]['user']}")
    print(f"Cleaned:  {df.iloc[i]['clean_text']}")

# --- Step 4: Train-Test Split ---
print("\n" + "=" * 80)
print("🔀 SPLITTING DATA")
print("=" * 80)

X = df["clean_text"]
y = df["intent"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train set size: {len(X_train)}")
print(f"✅ Test set size: {len(X_test)}")
print(f"✅ Train/Test ratio: 80/20")

# --- Step 5: Vectorize + Train Model ---
print("\n" + "=" * 80)
print("🤖 TRAINING INTENT CLASSIFIER")
print("=" * 80)

print("\n📝 Creating TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
Xv_train = vectorizer.fit_transform(X_train)
Xv_test = vectorizer.transform(X_test)
print(f"✅ Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"✅ Feature matrix shape: {Xv_train.shape}")

print("\n🎓 Training Logistic Regression model...")
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(Xv_train, y_train)
print("✅ Model training completed")

print("\n🔍 Making predictions...")
preds_train = model.predict(Xv_train)
preds_test = model.predict(Xv_test)

# --- Step 6: Evaluation ---
print("\n" + "=" * 80)
print("📊 MODEL EVALUATION")
print("=" * 80)

train_acc = accuracy_score(y_train, preds_train)
test_acc = accuracy_score(y_test, preds_test)

print(f"\n🎯 Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"🎯 Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

print("\n" + "-" * 80)
print("CLASSIFICATION REPORT (Test Set)")
print("-" * 80)
report = classification_report(y_test, preds_test)
print(report)

# Save report to file
with open("report.txt", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("MULTILINGUAL BANKING CHATBOT - CLASSIFICATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
    f.write(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
    f.write("-" * 80 + "\n")
    f.write("CLASSIFICATION REPORT (Test Set)\n")
    f.write("-" * 80 + "\n")
    f.write(report)
print("\n✅ Report saved to: report.txt")

# --- Step 7: Visualizations ---
print("\n" + "=" * 80)
print("📈 CREATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# 1. Intent Distribution
print("\n📊 Creating intent distribution plot...")
plt.figure(figsize=(12, 6))
intent_counts = df['intent'].value_counts()
colors = plt.cm.Set3(range(len(intent_counts)))
intent_counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=0.5)
plt.title('Intent Distribution in Banking FAQ Dataset', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Intent', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('intent_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: intent_distribution.png")

# 2. Confusion Matrix
print("\n📊 Creating confusion matrix...")
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, preds_test)
# Get class labels
labels = sorted(model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
plt.title('Confusion Matrix - Intent Classification', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Intent', fontsize=12, fontweight='bold')
plt.ylabel('Actual Intent', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")

# 3. Language Distribution
print("\n📊 Creating language distribution plot...")
plt.figure(figsize=(8, 6))
lang_counts = df['lang'].value_counts()
colors_lang = ['#3498db', '#e74c3c']
plt.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors_lang, explode=(0.05, 0.05),
        textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Language Distribution', fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')
plt.tight_layout()
plt.savefig('language_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: language_distribution.png")

# Close all figures to free memory
plt.close('all')

# --- Step 8: Chatbot Functions ---
print("\n" + "=" * 80)
print("🤖 BUILDING CHATBOT SYSTEM")
print("=" * 80)

def detect_language(text):
    """
    Detect if text contains Tamil characters.
    Returns 'Tamil' if Tamil characters found, otherwise 'English'.
    """
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "Tamil"
    return "English"

def chatbot_response(user_input, show_details=True):
    """
    Process user input and generate chatbot response.
    
    Steps:
    1. Detect language
    2. Clean and normalize input
    3. Predict intent using trained model
    4. Retrieve appropriate bot response from dataset
    5. Format and return response
    """
    # Detect language
    lang = detect_language(user_input)
    
    # Clean input
    cleaned = clean_text(user_input)
    
    # Predict intent
    pred_intent = model.predict(vectorizer.transform([cleaned]))[0]
    
    # Get confidence scores
    proba = model.predict_proba(vectorizer.transform([cleaned]))[0]
    confidence = max(proba) * 100
    
    # Retrieve bot response from dataset (prefer same language)
    subset = df[(df['intent'] == pred_intent) & (df['lang'] == lang)]
    if subset.empty:
        # Fallback to any language for that intent
        subset = df[df['intent'] == pred_intent]
    
    if not subset.empty:
        reply = subset.sample(1).iloc[0]["bot"]
    else:
        reply = "I'm here to help with your banking query. Please provide more details."
    
    if show_details:
        return f"🌐 Language: {lang} | 🎯 Intent: {pred_intent} | 📊 Confidence: {confidence:.1f}%\n💬 Bot: {reply}"
    else:
        return reply

print("✅ Chatbot system initialized")

# --- Step 9: Chatbot Demo ---
print("\n" + "=" * 80)
print("🧪 TESTING CHATBOT WITH SAMPLE QUERIES")
print("=" * 80)

# English test queries
english_queries = [
    "I want to open a new savings account",
    "How can I check my account balance?",
    "My debit card has been lost",
    "Need to update my KYC details",
    "What is the current FD interest rate?"
]

# Tamil test queries
tamil_queries = [
    "எனக்கு ஒரு புதிய சேமிப்பு கணக்கு திறக்க வேண்டும்",
    "என் கணக்கு இருப்பு எவ்வளவு?",
    "என் டெபிட் கார்டு தொலைந்துவிட்டது",
    "என் KYC விவரங்களை புதுப்பிக்க வேண்டும்",
    "நிலையான வைப்பு வட்டி விகிதம் என்ன?"
]

print("\n" + "🔹" * 40)
print("ENGLISH QUERIES")
print("🔹" * 40)

for i, query in enumerate(english_queries, 1):
    print(f"\n[Query {i}]")
    print(f"👤 User: {query}")
    response = chatbot_response(query)
    print(response)
    print("-" * 80)

print("\n" + "🔹" * 40)
print("தமிழ் கேள்விகள் (TAMIL QUERIES)")
print("🔹" * 40)

for i, query in enumerate(tamil_queries, 1):
    print(f"\n[Query {i}]")
    print(f"👤 User: {query}")
    response = chatbot_response(query)
    print(response)
    print("-" * 80)

# --- Step 10: Save Models ---
print("\n" + "=" * 80)
print("💾 SAVING MODELS AND ARTIFACTS")
print("=" * 80)

# Save model
with open("intent_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Saved: intent_model.pkl")

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ Saved: vectorizer.pkl")

# Save dataset info
with open("dataset_info.txt", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("MULTILINGUAL BANKING CHATBOT - DATASET INFORMATION\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total Records: {len(df)}\n")
    f.write(f"English Records: {len(eng_df)}\n")
    f.write(f"Tamil Records: {len(tam_df)}\n")
    f.write(f"Unique Intents: {df['intent'].nunique()}\n\n")
    f.write("Intent Categories:\n")
    for intent in sorted(df['intent'].unique()):
        count = len(df[df['intent'] == intent])
        f.write(f"  - {intent}: {count} records\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("INTENT DISTRIBUTION\n")
    f.write("=" * 80 + "\n")
    f.write(str(df['intent'].value_counts()))
print("✅ Saved: dataset_info.txt")

# --- Final Summary ---
print("\n" + "=" * 80)
print("✨ PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\n📦 Generated Files:")
files = [
    "intent_model.pkl - Trained Logistic Regression model",
    "vectorizer.pkl - TF-IDF vectorizer",
    "report.txt - Classification report and metrics",
    "dataset_info.txt - Dataset statistics and information",
    "intent_distribution.png - Intent frequency visualization",
    "confusion_matrix.png - Model performance heatmap",
    "language_distribution.png - Language distribution pie chart"
]
for file in files:
    print(f"  ✓ {file}")

print("\n📊 Model Performance Summary:")
print(f"  • Train Accuracy: {train_acc*100:.2f}%")
print(f"  • Test Accuracy:  {test_acc*100:.2f}%")
print(f"  • Total Intents:  {df['intent'].nunique()}")
print(f"  • Vocabulary Size: {len(vectorizer.vocabulary_)}")

print("\n🎯 Capabilities:")
print("  ✓ Multilingual support (English & Tamil)")
print("  ✓ Automatic language detection")
print("  ✓ Intent classification")
print("  ✓ Context-aware responses")
print("  ✓ Confidence scoring")

print("\n" + "=" * 80)
print("Thank you for using the Multilingual Banking Chatbot!")
print("=" * 80)
