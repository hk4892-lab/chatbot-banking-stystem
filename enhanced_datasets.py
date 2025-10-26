#!/usr/bin/env python3
"""
Enhanced Dataset Generator for Multilingual Banking Chatbot
This script creates larger datasets with more samples per intent for better model performance.
"""

import pandas as pd
import csv

# Enhanced English Banking FAQ Dataset
english_data = [
    # Account Opening (5 samples)
    ("I want to open a new account", "You can open a new account by visiting our nearest branch with valid ID proof and address proof. Our team will guide you through the process.", "account_opening"),
    ("How can I create a bank account", "To create a bank account, please bring your ID, address proof, and initial deposit to any of our branches.", "account_opening"),
    ("What do I need to open a savings account", "For opening a savings account, you need valid identification, address verification, and minimum deposit amount.", "account_opening"),
    ("I would like to start banking with you", "We welcome you to start banking with us. Visit our branch with required documents to open your account.", "account_opening"),
    ("Can you help me open an account", "Certainly! We can help you open an account. Please visit our branch with necessary documents.", "account_opening"),
    
    # Balance Inquiry (5 samples)
    ("How do I check my account balance", "You can check your account balance through our mobile app, internet banking, or by visiting an ATM.", "balance_inquiry"),
    ("What is my current balance", "You can view your current balance through our digital banking services or ATM machines.", "balance_inquiry"),
    ("How to see my account balance", "Your account balance is available 24/7 through our mobile app, online banking, or ATM network.", "balance_inquiry"),
    ("Can I check my balance online", "Yes, you can check your balance online through our internet banking portal or mobile application.", "balance_inquiry"),
    ("Where can I see my account balance", "You can see your account balance through multiple channels: mobile app, internet banking, ATM, or branch visit.", "balance_inquiry"),
    
    # Card Services (5 samples)
    ("I lost my debit card", "Please report your lost debit card immediately by calling our 24/7 helpline. We will block your card and issue a new one.", "card_services"),
    ("My credit card is stolen", "Report your stolen credit card immediately. We will block it and issue a replacement within 3-5 business days.", "card_services"),
    ("I need to block my card", "You can block your card instantly by calling our customer service or using our mobile app.", "card_services"),
    ("My ATM card is not working", "If your ATM card is not working, please visit our branch or call customer service for assistance.", "card_services"),
    ("How to replace my damaged card", "For a damaged card replacement, visit our branch with ID proof and we will issue a new card.", "card_services"),
    
    # Loan Inquiry (5 samples)
    ("What are the loan interest rates", "Our loan interest rates vary by type. Personal loans start from 10.5% and home loans from 8.5%.", "loan_inquiry"),
    ("I want to apply for a personal loan", "You can apply for a personal loan online or at our branch with income and identity documents.", "loan_inquiry"),
    ("What is the home loan rate", "Our home loan rates start from 8.5% and vary based on loan amount and tenure.", "loan_inquiry"),
    ("How much loan can I get", "Your loan eligibility depends on your income, credit score, and repayment capacity.", "loan_inquiry"),
    ("What are the loan requirements", "Loan requirements include income proof, identity documents, and good credit history.", "loan_inquiry"),
    
    # KYC Update (5 samples)
    ("How can I update my KYC details", "You can update your KYC details by visiting our branch with updated documents or through our online portal.", "kyc_update"),
    ("I need to update my address", "To update your address, visit our branch with address proof or use our digital KYC update service.", "kyc_update"),
    ("How to change my phone number", "You can change your registered phone number by visiting our branch with ID proof.", "kyc_update"),
    ("Update my personal information", "Personal information can be updated at our branch or through our secure online portal.", "kyc_update"),
    ("I want to update my documents", "Document updates can be done by visiting our branch or uploading through our digital platform.", "kyc_update"),
]

# Enhanced Tamil Banking FAQ Dataset  
tamil_data = [
    # Account Opening (5 samples)
    ("எனக்கு ஒரு புதிய கணக்கு திறக்க வேண்டும்", "நீங்கள் செல்லுபடியாகும் அடையாள ஆதாரம் மற்றும் முகவரி ஆதாரத்துடன் எங்கள் அருகிலுள்ள கிளைக்குச் சென்று புதிய கணக்கைத் திறக்கலாம்.", "account_opening"),
    ("வங்கி கணக்கு எப்படி திறக்கிறது", "வங்கி கணக்கு திறக்க அடையாள ஆதாரம், முகவரி ஆதாரம் மற்றும் ஆரம்ப வைப்புத் தொகையுடன் எங்கள் கிளைக்கு வாருங்கள்.", "account_opening"),
    ("சேமிப்பு கணக்கு திறக்க என்ன வேண்டும்", "சேமிப்பு கணக்கு திறக்க செல்லுபடியாகும் அடையாளம், முகவரி சரிபார்ப்பு மற்றும் குறைந்தபட்ச வைப்புத் தொகை தேவை.", "account_opening"),
    ("உங்களுடன் வங்கி பணி தொடங்க விரும்புகிறேன்", "எங்களுடன் வங்கி பணி தொடங்க உங்களை வரவேற்கிறோம். தேவையான ஆவணங்களுடன் எங்கள் கிளைக்கு வாருங்கள்.", "account_opening"),
    ("கணக்கு திறக்க உதவி செய்வீர்களா", "நிச்சயமாக! கணக்கு திறக்க உதவி செய்வோம். தேவையான ஆவணங்களுடன் எங்கள் கிளைக்கு வாருங்கள்.", "account_opening"),
    
    # Balance Inquiry (5 samples)
    ("என் கணக்கு இருப்பு எப்படி பார்க்கிறது", "நீங்கள் எங்கள் மொபைல் ஆப், இணைய வங்கி அல்லது ஏடிஎம் மூலம் உங்கள் கணக்கு இருப்பைப் பார்க்கலாம்.", "balance_inquiry"),
    ("என் தற்போதைய இருப்பு என்ன", "உங்கள் தற்போதைய இருப்பை எங்கள் டிஜிட்டல் வங்கி சேவைகள் அல்லது ஏடிஎம் மூலம் பார்க்கலாம்.", "balance_inquiry"),
    ("கணக்கு இருப்பை எப்படி பார்ப்பது", "உங்கள் கணக்கு இருப்பு 24/7 எங்கள் மொபைல் ஆப், ஆன்லைன் வங்கி அல்லது ஏடிஎம் நெட்வொர்க் மூலம் கிடைக்கும்.", "balance_inquiry"),
    ("ஆன்லайனில் இருப்பு பார்க்க முடியுமா", "ஆம், எங்கள் இணைய வங்கி போர்ட்டல் அல்லது மொபைல் ஆப்ளிகேஷன் மூலம் ஆன்லைனில் இருப்பு பார்க்கலாம்.", "balance_inquiry"),
    ("கணக்கு இருப்பை எங்கே பார்க்கலாம்", "கணக்கு இருப்பை பல வழிகளில் பார்க்கலாம்: மொபைல் ஆப், இணைய வங்கி, ஏடிஎம் அல்லது கிளை வருகை.", "balance_inquiry"),
    
    # Card Services (5 samples)
    ("என் டெபிட் கார்டு தொலைந்துவிட்டது", "உங்கள் தொலைந்த டெபிட் கார்டை உடனடியாக எங்கள் 24/7 உதவி எண்ணில் தெரிவிக்கவும். நாங்கள் உங்கள் கார்டைத் தடுத்து புதியதை வழங்குவோம்.", "card_services"),
    ("என் கிரெடிட் கார்டு திருடப்பட்டது", "திருடப்பட்ட கிரெடிட் கார்டை உடனடியாக தெரிவிக்கவும். நாங்கள் அதைத் தடுத்து 3-5 வணிக நாட்களில் மாற்றீடு வழங்குவோம்.", "card_services"),
    ("என் கார்டை தடுக்க வேண்டும்", "எங்கள் வாடிக்கையாளர் சேவையை அழைத்து அல்லது எங்கள் மொபைல் ஆப் பயன்படுத்தி உடனடியாக கார்டைத் தடுக்கலாம்.", "card_services"),
    ("என் ஏடிஎம் கார்டு வேலை செய்யவில்லை", "உங்கள் ஏடிஎம் கார்டு வேலை செய்யவில்லை என்றால், எங்கள் கிளைக்கு வாருங்கள் அல்லது வாடிக்கையாளர் சேவையை அழைக்கவும்.", "card_services"),
    ("சேதமடைந்த கார்டை எப்படி மாற்றுவது", "சேதமடைந்த கார்டு மாற்றத்திற்கு அடையாள ஆதாரத்துடன் எங்கள் கிளைக்கு வந்து புதிய கார்டு பெறலாம்.", "card_services"),
    
    # Loan Inquiry (5 samples)
    ("கடன் வட்டி விகிதங்கள் என்ன", "எங்கள் கடன் வட்டி விகிதங்கள் வகையைப் பொறுத்து மாறுபடும். தனிநபர் கடன்கள் 10.5% முதல் வீட்டுக் கடன்கள் 8.5% முதல்.", "loan_inquiry"),
    ("தனிநபர் கடனுக்கு விண்ணப்பிக்க விரும்புகிறேன்", "வருமானம் மற்றும் அடையாள ஆவணங்களுடன் ஆன்லைன் அல்லது எங்கள் கிளையில் தனிநபர் கடனுக்கு விண்ணப்பிக்கலாம்.", "loan_inquiry"),
    ("வீட்டுக் கடன் விகிதம் என்ன", "எங்கள் வீட்டுக் கடன் விகிதங்கள் 8.5% முதல் தொடங்கி கடன் தொகை மற்றும் காலத்தைப் பொறுத்து மாறுபடும்.", "loan_inquiry"),
    ("எனக்கு எவ்வளவு கடன் கிடைக்கும்", "உங்கள் கடன் தகுதி உங்கள் வருமானம், கிரெடிட் ஸ்கோர் மற்றும் திருப்பிச் செலுத்தும் திறனைப் பொறுத்தது.", "loan_inquiry"),
    ("கடன் தேவைகள் என்ன", "கடன் தேவைகளில் வருமான ஆதாரம், அடையாள ஆவணங்கள் மற்றும் நல்ல கிரெடிட் வரலாறு அடங்கும்.", "loan_inquiry"),
    
    # KYC Update (5 samples)
    ("என் KYC விவரங்களை எப்படி புதுப்பிக்கிறது", "புதுப்பிக்கப்பட்ட ஆவணங்களுடன் எங்கள் கிளைக்குச் சென்று அல்லது எங்கள் ஆன்லைன் போர்ட்டல் மூலம் KYC விவரங்களைப் புதுப்பிக்கலாம்.", "kyc_update"),
    ("என் முகவரியை புதுப்பிக்க வேண்டும்", "முகவரியைப் புதுப்பிக்க முகவரி ஆதாரத்துடன் எங்கள் கிளைக்கு வாருங்கள் அல்லது எங்கள் டிஜிட்டல் KYC புதுப்பிப்பு சேவையைப் பயன்படுத்தவும்.", "kyc_update"),
    ("என் தொலைபேசி எண்ணை எப்படி மாற்றுவது", "பதிவு செய்யப்பட்ட தொலைபேசி எண்ணை அடையாள ஆதாரத்துடன் எங்கள் கிளைக்கு வந்து மாற்றலாம்.", "kyc_update"),
    ("என் தனிப்பட்ட தகவல்களை புதுப்பிக்கவும்", "தனிப்பட்ட தகவல்களை எங்கள் கிளையில் அல்லது எங்கள் பாதுகாப்பான ஆன்லைன் போர்ட்டல் மூலம் புதுப்பிக்கலாம்.", "kyc_update"),
    ("என் ஆவணங்களை புதுப்பிக்க விரும்புகிறேன்", "ஆவண புதுப்பிப்புகளை எங்கள் கிளைக்கு வந்து அல்லது எங்கள் டிஜிட்டல் தளத்தில் பதிவேற்றம் செய்து செய்யலாம்.", "kyc_update"),
]

def create_enhanced_datasets():
    """Create enhanced datasets with more samples per intent"""
    
    # Create English dataset
    with open('/workspace/BankFAQs_Enhanced.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'bot', 'intent'])
        writer.writerows(english_data)
    
    # Create Tamil dataset
    with open('/workspace/BankFAQs_Tamil_Enhanced.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'bot', 'intent'])
        writer.writerows(tamil_data)
    
    print("✅ Enhanced datasets created successfully!")
    print(f"📊 English samples: {len(english_data)}")
    print(f"📊 Tamil samples: {len(tamil_data)}")
    print(f"📊 Total samples: {len(english_data) + len(tamil_data)}")
    
    # Display sample data
    print("\n📋 Sample English Data:")
    for i in range(3):
        print(f"User: {english_data[i][0]}")
        print(f"Bot: {english_data[i][1]}")
        print(f"Intent: {english_data[i][2]}")
        print("-" * 50)
    
    print("\n📋 Sample Tamil Data:")
    for i in range(3):
        print(f"User: {tamil_data[i][0]}")
        print(f"Bot: {tamil_data[i][1]}")
        print(f"Intent: {tamil_data[i][2]}")
        print("-" * 50)

if __name__ == "__main__":
    create_enhanced_datasets()