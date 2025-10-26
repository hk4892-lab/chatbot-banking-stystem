#!/usr/bin/env python3
"""
Demo Script for Multilingual Banking Chatbot
This script demonstrates how to use the trained models for real-time chatbot interactions.

Author: ML Engineer
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

class MultilingualBankingChatbot:
    """
    Production-ready Multilingual Banking Chatbot
    """
    
    def __init__(self, model_path="enhanced_intent_classifier.pkl", 
                 vectorizer_path="enhanced_tfidf_vectorizer.pkl",
                 dataset_path="enhanced_processed_dataset.csv"):
        """
        Initialize the chatbot with trained models
        """
        print("🤖 Initializing Multilingual Banking Chatbot...")
        
        # Load trained models
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Intent classifier loaded from {model_path}")
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"✅ TF-IDF vectorizer loaded from {vectorizer_path}")
            
            # Load dataset for responses
            self.df = pd.read_csv(dataset_path)
            print(f"✅ Response dataset loaded from {dataset_path}")
            
            print(f"🎯 Ready to handle {len(self.model.classes_)} intent categories")
            print(f"🌐 Supporting English and Tamil languages")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def clean_text(self, text):
        """Clean and normalize text for both English and Tamil"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def detect_language(self, text):
        """Detect if text is Tamil or English"""
        if not text or pd.isna(text):
            return "Unknown"
        
        text = str(text)
        tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
        total_chars = len(re.findall(r'[^\s\W]', text))
        
        if total_chars == 0:
            return "Unknown"
        
        tamil_ratio = tamil_chars / total_chars
        return "Tamil" if tamil_ratio > 0.1 else "English"
    
    def get_language_confidence(self, text):
        """Get confidence score for language detection"""
        if not text or pd.isna(text):
            return 0.0
        
        text = str(text)
        tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
        total_chars = len(re.findall(r'[^\s\W]', text))
        
        if total_chars == 0:
            return 0.0
        
        tamil_ratio = tamil_chars / total_chars
        
        if tamil_ratio > 0.1:
            return min(tamil_ratio * 2, 1.0)
        else:
            return min((1 - tamil_ratio) * 1.2, 1.0)
    
    def predict_intent(self, user_input):
        """Predict intent from user input"""
        cleaned_input = self.clean_text(user_input)
        
        if not cleaned_input.strip():
            return "unknown", 0.0, []
        
        # Predict intent
        pred_intent = self.model.predict(self.vectorizer.transform([cleaned_input]))[0]
        intent_proba = self.model.predict_proba(self.vectorizer.transform([cleaned_input]))[0]
        intent_confidence = intent_proba.max()
        
        # Get top 3 predictions
        top_indices = np.argsort(intent_proba)[-3:][::-1]
        top_intents = [(self.model.classes_[i], intent_proba[i]) for i in top_indices]
        
        return pred_intent, intent_confidence, top_intents
    
    def get_response(self, user_input):
        """Generate complete chatbot response"""
        # Language detection
        detected_lang = self.detect_language(user_input)
        lang_confidence = self.get_language_confidence(user_input)
        
        # Intent prediction
        pred_intent, intent_confidence, top_intents = self.predict_intent(user_input)
        
        # Find appropriate response
        subset = self.df[(self.df['intent'] == pred_intent) & (self.df['lang'] == detected_lang)]
        
        if subset.empty:
            subset = self.df[self.df['intent'] == pred_intent]
        
        if not subset.empty:
            selected_row = subset.sample(1).iloc[0]
            bot_reply = selected_row['bot']
            response_lang = selected_row['lang']
        else:
            # Fallback responses
            if detected_lang == "Tamil":
                bot_reply = "மன்னிக்கவும், உங்கள் கேள்வியை நான் புரிந்து கொள்ளவில்லை. தயவுசெய்து மற்றொரு வழியில் கேட்கவும்."
            else:
                bot_reply = "I'm sorry, I didn't understand your question. Please try asking in a different way."
            response_lang = detected_lang
        
        return {
            'user_input': user_input,
            'detected_language': detected_lang,
            'language_confidence': lang_confidence,
            'predicted_intent': pred_intent,
            'intent_confidence': intent_confidence,
            'top_predictions': top_intents,
            'response': bot_reply,
            'response_language': response_lang
        }
    
    def chat(self, user_input):
        """Main chat interface"""
        response_data = self.get_response(user_input)
        
        print(f"\n👤 You ({response_data['detected_language']}): {user_input}")
        print(f"🤖 Banking Assistant:")
        print(f"   🌐 Language: {response_data['detected_language']} (Confidence: {response_data['language_confidence']:.2f})")
        print(f"   🎯 Intent: {response_data['predicted_intent']} (Confidence: {response_data['intent_confidence']:.2f})")
        
        if response_data['top_predictions']:
            print(f"   📊 Top Predictions:")
            for i, (intent, prob) in enumerate(response_data['top_predictions'][:3]):
                print(f"      {i+1}. {intent}: {prob:.2f}")
        
        print(f"   💬 Response: {response_data['response']}")
        
        return response_data

def run_interactive_demo():
    """Run interactive chatbot demo"""
    print("🚀 Multilingual Banking Chatbot - Interactive Demo")
    print("=" * 60)
    
    try:
        # Initialize chatbot
        chatbot = MultilingualBankingChatbot()
        
        print("\n🎉 Chatbot initialized successfully!")
        print("\n📋 Available Intent Categories:")
        for i, intent in enumerate(chatbot.model.classes_, 1):
            print(f"   {i}. {intent}")
        
        print("\n💡 Sample Queries:")
        print("   English: 'I want to open an account', 'Check my balance'")
        print("   Tamil: 'எனக்கு கணக்கு திறக்க வேண்டும்', 'என் இருப்பு பார்க்க வேண்டும்'")
        print("\n🔄 Type 'quit' or 'exit' to end the conversation")
        print("=" * 60)
        
        # Interactive loop
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n🤖 Banking Assistant: Thank you for using our service! வணக்கம்!")
                break
            
            if not user_input:
                print("🤖 Banking Assistant: Please enter your query.")
                continue
            
            # Get chatbot response
            chatbot.chat(user_input)
            
    except KeyboardInterrupt:
        print("\n\n🤖 Banking Assistant: Goodbye! Thanks for using our service!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def run_batch_demo():
    """Run batch demo with predefined queries"""
    print("🚀 Multilingual Banking Chatbot - Batch Demo")
    print("=" * 60)
    
    try:
        # Initialize chatbot
        chatbot = MultilingualBankingChatbot()
        
        # Demo queries
        demo_queries = [
            "I want to open a new account",
            "How do I check my balance?",
            "I lost my card",
            "What are the loan rates?",
            "Update my KYC details",
            "எனக்கு புதிய கணக்கு வேண்டும்",
            "என் இருப்பு எப்படி பார்க்கிறது?",
            "என் கார்டு தொலைந்துவிட்டது",
            "கடன் வட்டி என்ன?",
            "KYC புதுப்பிக்க வேண்டும்"
        ]
        
        print(f"\n🧪 Running batch demo with {len(demo_queries)} queries...")
        print("=" * 60)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n--- Demo Query {i} ---")
            chatbot.chat(query)
            
        print("\n" + "=" * 60)
        print("🎉 Batch demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    print("🤖 Multilingual Banking Chatbot Demo")
    print("Choose demo mode:")
    print("1. Interactive Demo (chat with the bot)")
    print("2. Batch Demo (predefined queries)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            run_interactive_demo()
            break
        elif choice == "2":
            run_batch_demo()
            break
        else:
            print("Please enter 1 or 2")