"""
Interactive Demo Script for Multilingual Banking Chatbot

This script loads the trained model and provides an interactive interface
to test the chatbot with custom queries.
"""

import pickle
import re
import pandas as pd

# Load trained model and vectorizer
print("Loading trained models...")
model = pickle.load(open("intent_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load and prepare dataframes
eng_df = pd.read_csv("BankFAQs.csv")
tam_df = pd.read_csv("BankFAQs_Tamil.csv")
eng_df["lang"] = "English"
tam_df["lang"] = "Tamil"
df = pd.concat([eng_df, tam_df], ignore_index=True)
print("✅ Models loaded successfully!\n")

def clean_text(text):
    """Clean and normalize text for prediction"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text):
    """Detect if text is English or Tamil"""
    return "Tamil" if re.search(r'[\u0B80-\u0BFF]', text) else "English"

def chatbot_response(user_input):
    """Process user input and generate response"""
    # Detect language
    lang = detect_language(user_input)
    
    # Clean input
    cleaned = clean_text(user_input)
    
    # Predict intent
    pred_intent = model.predict(vectorizer.transform([cleaned]))[0]
    
    # Get confidence scores
    proba = model.predict_proba(vectorizer.transform([cleaned]))[0]
    confidence = max(proba) * 100
    
    # Retrieve bot response (prefer same language)
    subset = df[(df['intent'] == pred_intent) & (df['lang'] == lang)]
    if subset.empty:
        subset = df[df['intent'] == pred_intent]
    
    if not subset.empty:
        reply = subset.sample(1).iloc[0]["bot"]
    else:
        reply = "I'm here to help with your banking query. Please provide more details."
    
    return lang, pred_intent, confidence, reply

# Print header
print("=" * 80)
print("🤖 MULTILINGUAL BANKING CHATBOT - Interactive Demo")
print("=" * 80)
print("\nSupported Languages: English, Tamil (தமிழ்)")
print("Supported Intents: 13 banking categories")
print("\nType 'quit' or 'exit' to end the session")
print("Type 'examples' to see sample queries")
print("=" * 80 + "\n")

# Interactive loop
while True:
    user_input = input("👤 You: ").strip()
    
    if not user_input:
        continue
    
    if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
        print("\n👋 Thank you for using the Banking Chatbot! Goodbye!")
        break
    
    if user_input.lower() == 'examples':
        print("\n" + "=" * 80)
        print("📝 SAMPLE QUERIES")
        print("=" * 80)
        print("\n🔹 English Examples:")
        print("  • How do I open a new account?")
        print("  • Check my balance")
        print("  • I lost my card")
        print("  • What are the interest rates?")
        print("  • How to transfer money?")
        print("\n🔹 Tamil Examples (தமிழ்):")
        print("  • புதிய கணக்கு திறக்க வேண்டும்")
        print("  • என் இருப்பு என்ன?")
        print("  • கார்டு தொலைந்துவிட்டது")
        print("  • வட்டி விகிதம் என்ன?")
        print("  • பணம் அனுப்ப எப்படி?")
        print("=" * 80 + "\n")
        continue
    
    # Get chatbot response
    try:
        lang, intent, confidence, reply = chatbot_response(user_input)
        
        print(f"\n{'─' * 80}")
        print(f"🌐 Language:  {lang}")
        print(f"🎯 Intent:    {intent}")
        print(f"📊 Confidence: {confidence:.1f}%")
        print(f"{'─' * 80}")
        print(f"🤖 Bot: {reply}")
        print(f"{'─' * 80}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please try another query.\n")

print("\n" + "=" * 80)
print("Session ended. Check report.txt for model performance metrics.")
print("=" * 80)
