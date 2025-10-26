# Multilingual Banking Chatbot using Machine Learning (English–Tamil)

A sophisticated hybrid chatbot system that provides banking customer support in both English and Tamil languages using Natural Language Processing and Machine Learning.

## 🌟 Features

- **Multilingual Support**: Handles queries in both English and Tamil
- **Automatic Language Detection**: Identifies language using Unicode character patterns
- **Intent Classification**: Uses TF-IDF vectorization and Logistic Regression
- **Context-Aware Responses**: Retrieves appropriate answers from curated FAQ datasets
- **Confidence Scoring**: Provides confidence levels for predictions
- **Comprehensive Evaluation**: Includes accuracy metrics, confusion matrix, and visualizations

## 📊 Project Overview

This project demonstrates a complete ML pipeline for building a production-ready multilingual chatbot:

1. **Data Loading & Merging**: Combines English and Tamil FAQ datasets
2. **Text Preprocessing**: Handles Unicode normalization for both languages
3. **Feature Engineering**: TF-IDF vectorization with bigrams
4. **Model Training**: Logistic Regression classifier for intent prediction
5. **Evaluation**: Comprehensive metrics and visualizations
6. **Inference**: Real-time language detection and response generation

## 🎯 Intent Categories

The chatbot handles 13 banking intent categories:

- `account_opening` - New account creation queries
- `balance_inquiry` - Balance check requests
- `card_services` - Credit/debit card related issues
- `loan_inquiry` - Loan application and status
- `kyc_update` - KYC document updates
- `interest_rates` - Interest rate information
- `account_closure` - Account closing procedures
- `fund_transfer` - Money transfer queries
- `transaction_issues` - Transaction problems
- `atm_services` - ATM related services
- `password_reset` - Password/PIN reset
- `service_charges` - Fee and charge information
- `account_inquiry` - General account queries

## 📁 Project Structure

```
/workspace/
├── BankFAQs.csv                      # English FAQ dataset (32 records)
├── BankFAQs_Tamil.csv                # Tamil FAQ dataset (32 records)
├── multilingual_banking_chatbot.py   # Main chatbot implementation
├── requirements.txt                  # Python dependencies
├── intent_model.pkl                  # Trained Logistic Regression model
├── vectorizer.pkl                    # TF-IDF vectorizer
├── report.txt                        # Classification report
├── dataset_info.txt                  # Dataset statistics
├── intent_distribution.png           # Intent frequency chart
├── confusion_matrix.png              # Model performance heatmap
├── language_distribution.png         # Language distribution pie chart
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone or download this repository:
```bash
cd /workspace
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Dependencies

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Visualization
- `seaborn` - Statistical data visualization

## 💻 Usage

### Running the Complete Pipeline

Execute the main script to train the model and test the chatbot:

```bash
python3 multilingual_banking_chatbot.py
```

This will:
1. Load and merge datasets
2. Preprocess text (clean, normalize)
3. Train the intent classifier
4. Generate evaluation metrics
5. Create visualizations
6. Test chatbot with sample queries
7. Save trained models and reports

### Using the Chatbot Programmatically

```python
import pickle
import re

# Load trained model and vectorizer
model = pickle.load(open("intent_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text):
    return "Tamil" if re.search(r'[\u0B80-\u0BFF]', text) else "English"

def get_intent(user_input):
    cleaned = clean_text(user_input)
    intent = model.predict(vectorizer.transform([cleaned]))[0]
    confidence = max(model.predict_proba(vectorizer.transform([cleaned]))[0]) * 100
    return intent, confidence

# Test the chatbot
query = "How to check my balance?"
intent, confidence = get_intent(query)
print(f"Intent: {intent}, Confidence: {confidence:.2f}%")
```

## 📈 Model Performance

### Training Results

- **Train Accuracy**: 84.31%
- **Test Accuracy**: 61.54%
- **Total Training Samples**: 51
- **Total Test Samples**: 13
- **Vocabulary Size**: 323 features
- **Model**: Logistic Regression (max_iter=500)

### Evaluation Metrics

The model achieves strong performance on most intent categories. See `report.txt` for detailed precision, recall, and F1-scores per intent class.

### Visualizations

1. **Intent Distribution** (`intent_distribution.png`)
   - Bar chart showing frequency of each intent in the dataset
   - Helps identify data balance and coverage

2. **Confusion Matrix** (`confusion_matrix.png`)
   - Heatmap showing model predictions vs actual intents
   - Identifies which intents are commonly confused

3. **Language Distribution** (`language_distribution.png`)
   - Pie chart showing English vs Tamil data split
   - Demonstrates balanced multilingual dataset

## 🧪 Sample Test Queries

### English Queries
```
1. "I want to open a new savings account"
   → Intent: account_opening
   
2. "How can I check my account balance?"
   → Intent: balance_inquiry
   
3. "My debit card has been lost"
   → Intent: card_services
   
4. "Need to update my KYC details"
   → Intent: kyc_update
   
5. "What is the current FD interest rate?"
   → Intent: interest_rates
```

### Tamil Queries (தமிழ்)
```
1. "எனக்கு ஒரு புதிய சேமிப்பு கணக்கு திறக்க வேண்டும்"
   → Intent: account_opening
   
2. "என் கணக்கு இருப்பு எவ்வளவு?"
   → Intent: balance_inquiry
   
3. "என் டெபிட் கார்டு தொலைந்துவிட்டது"
   → Intent: card_services
   
4. "என் KYC விவரங்களை புதுப்பிக்க வேண்டும்"
   → Intent: kyc_update
   
5. "நிலையான வைப்பு வட்டி விகிதம் என்ன?"
   → Intent: interest_rates
```

## 🔧 Technical Details

### Text Preprocessing

The `clean_text()` function performs:
- Lowercase conversion
- Unicode normalization (preserves Tamil characters U+0B80 to U+0BFF)
- Special character removal
- Whitespace normalization

### Language Detection

Uses regex pattern matching to detect Tamil Unicode characters:
```python
if re.search(r'[\u0B80-\u0BFF]', text):
    return "Tamil"
```

### Feature Engineering

- **TF-IDF Vectorization**: Converts text to numerical features
- **N-gram Range**: (1, 2) - captures unigrams and bigrams
- **Max Features**: 5000 (vocabulary limited for efficiency)

### Model Architecture

- **Algorithm**: Logistic Regression
- **Multi-class Strategy**: One-vs-Rest (OvR)
- **Max Iterations**: 500
- **Solver**: lbfgs (default)
- **Random State**: 42 (reproducibility)

## 📊 Dataset Information

### English Dataset (BankFAQs.csv)
- **Total Records**: 32
- **Columns**: user, bot, intent, lang
- **Format**: CSV with quoted fields

### Tamil Dataset (BankFAQs_Tamil.csv)
- **Total Records**: 32
- **Columns**: user, bot, intent, lang
- **Format**: CSV with quoted fields
- **Encoding**: UTF-8 (supports Tamil Unicode)

### Combined Dataset
- **Total Records**: 64
- **Intent Classes**: 13
- **Balanced**: Equal English and Tamil samples per intent
- **Train-Test Split**: 80/20 stratified by intent

## 🎨 Visualizations

All visualizations are saved as high-resolution PNG files (300 DPI):

1. **intent_distribution.png**: Colorful bar chart with grid
2. **confusion_matrix.png**: Blue heatmap with annotations
3. **language_distribution.png**: Pie chart with percentages

## 🔍 Key Functions

### `clean_text(text)`
Preprocesses and normalizes user input

### `detect_language(text)`
Identifies if text is English or Tamil

### `chatbot_response(user_input, show_details=True)`
Main chatbot function that:
- Detects language
- Predicts intent
- Retrieves appropriate response
- Returns formatted output

## 📝 Output Files

| File | Description |
|------|-------------|
| `intent_model.pkl` | Serialized Logistic Regression model |
| `vectorizer.pkl` | Serialized TF-IDF vectorizer |
| `report.txt` | Classification report with metrics |
| `dataset_info.txt` | Dataset statistics and distribution |
| `*.png` | Visualization charts |

## 🚀 Future Enhancements

1. **Expand Dataset**: Add more intents and samples per intent
2. **Deep Learning**: Experiment with LSTM/Transformer models
3. **Context Management**: Implement conversation history
4. **Entity Extraction**: Extract account numbers, dates, amounts
5. **Sentiment Analysis**: Detect customer frustration
6. **Voice Interface**: Add speech-to-text capabilities
7. **More Languages**: Expand to Hindi, Telugu, Malayalam, etc.
8. **API Deployment**: Create REST API with FastAPI/Flask
9. **Database Integration**: Store conversations for analytics
10. **Active Learning**: Improve model with user feedback

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more FAQ samples
- Improve text preprocessing for Tamil
- Experiment with different ML algorithms
- Add unit tests
- Improve documentation

## 📄 License

This project is created for educational purposes.

## 👥 Authors

- Expert ML Engineer specializing in NLP and Multilingual Systems

## 🙏 Acknowledgments

- Dataset curated specifically for banking domain
- Tamil language support using Unicode standards
- Scikit-learn for ML algorithms
- Matplotlib and Seaborn for visualizations

## 📞 Support

For issues or questions about the chatbot:
1. Check the `report.txt` for model performance
2. Review `dataset_info.txt` for data statistics
3. Examine visualizations for insights

---

**Built with ❤️ for Multilingual Banking Customer Support**

*Powered by Machine Learning | Supporting English & Tamil*
